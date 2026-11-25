"""Kiểm chứng detector INT8 TRƯỚC khi nạp lên ESP (không cần board).

So sánh trên tập ảnh có nhãn (mặc định 1.200 ảnh val):
  A. PyTorch float + decode Python      (mức trần)
  B. TFLite INT8   + decode Python      (mất mát do lượng tử)
  C. TFLite INT8   + decode C THẬT của firmware (main/logic.c biên dịch bằng gcc trên máy)
B và C phải gần như trùng nhau; nếu lệch là decode C sai.

Sau đó in bảng precision / recall / đếm đúng theo ngưỡng conf (để chọn CONF_THRESH cho vòng chung kết:
tiêu chí phụ là precision = số vật đúng / số vật model báo).

    D:/edgeai/edgeai/Scripts/python.exe tools/verify_det.py
    D:/edgeai/edgeai/Scripts/python.exe tools/verify_det.py --images path/anh_camera --n 200
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).parent))
from export_det_yolo import H_IN, ROOT, W_IN, load_rgb, torch_head_outputs  # noqa: E402
from tfl_utils import FW, MODELS  # noqa: E402

NC = 7
NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


# ------------------------------------------------------------ decode Python (giống main/logic.c)
def decode_py(outs, reg_max, conf=0.001, iou=0.7, agnostic=False, max_det=300, dist_scale_last=1.0):
    """outs: list [(box HxWx4R, cls HxWxNC)] float (đã dequant). Trả về N x 6 [cls, score, x1, y1, x2, y2] chuẩn hóa.
    dist_scale_last: hệ số tầm với mức cuối (tools/yolo_reach.py), giống YOLO_DIST_SCALE_LAST trong logic.h."""
    boxes, scores, labels = [], [], []
    for li, (b, c) in enumerate(outs):
        H, W = c.shape[:2]
        stride = H_IN / H
        best = c.argmax(-1)
        sc = 1 / (1 + np.exp(-c.max(-1)))
        ys, xs = np.nonzero(sc >= conf)
        if not len(ys):
            continue
        d = b[ys, xs].reshape(-1, 4, reg_max)
        d = np.exp(d - d.max(-1, keepdims=True))
        d = (d / d.sum(-1, keepdims=True) * np.arange(reg_max)).sum(-1)
        if li == len(outs) - 1:
            d = d * dist_scale_last
        ax, ay = xs + 0.5, ys + 0.5
        bx = np.stack([(ax - d[:, 0]) * stride / W_IN, (ay - d[:, 1]) * stride / H_IN,
                       (ax + d[:, 2]) * stride / W_IN, (ay + d[:, 3]) * stride / H_IN], 1).clip(0, 1)
        boxes.append(bx), scores.append(sc[ys, xs]), labels.append(best[ys, xs])
    if not boxes:
        return np.zeros((0, 6), np.float32)
    bx, sc, lb = (torch.from_numpy(np.concatenate(v)).float() for v in (boxes, scores, labels))
    keep = torchvision.ops.batched_nms(bx, sc, torch.zeros_like(lb) if agnostic else lb, iou)[:max_det]
    return torch.cat([lb[keep, None], sc[keep, None], bx[keep]], 1).numpy()


# ------------------------------------------------------------ metric
def load_gt(img: Path):
    lp = img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
    if not lp.exists():
        lp = Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt")
    if not lp.exists():
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(lp, ndmin=2).astype(np.float32)
    if not len(a):
        return np.zeros((0, 5), np.float32)
    c, x, y, w, h = a.T
    return np.stack([c, x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def iou_mat(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], -1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(pred, gt, thr=0.5):
    """tp flag cho từng box dự đoán (đã sort theo score giảm dần), ghép tham lam cùng lớp IoU >= thr."""
    tp = np.zeros(len(pred), bool)
    if not len(pred) or not len(gt):
        return tp
    ious = iou_mat(pred[:, 2:6], gt[:, 1:5])
    used = np.zeros(len(gt), bool)
    for i in np.argsort(-pred[:, 1]):
        cand = np.where((gt[:, 0] == pred[i, 0]) & ~used, ious[i], -1)
        j = cand.argmax()
        if cand[j] >= thr:
            tp[i], used[j] = True, True
    return tp


def map50(preds, gts):
    aps = []
    for c in range(NC):
        sc, tp, ngt = [], [], 0
        for p, g in zip(preds, gts):
            pc, gc = p[p[:, 0] == c], g[g[:, 0] == c]
            ngt += len(gc)
            sc.append(pc[:, 1]), tp.append(match(pc, gc))
        sc, tp = np.concatenate(sc), np.concatenate(tp)
        if not ngt:
            continue
        o = np.argsort(-sc)
        ctp = np.cumsum(tp[o])
        rec, prec = ctp / ngt, ctp / np.arange(1, len(o) + 1)
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        ap = np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)])
        aps.append(ap)
    return float(np.mean(aps)), aps


def threshold_table(preds, gts, confs):
    print(f"\n{'conf':>5} {'precision':>9} {'recall':>7} {'F1':>6} {'đếm đúng/ảnh':>13} {'box sai/ảnh':>11}")
    rows = []
    for t in confs:
        tp = fp = ngt = cnt_ok = 0
        for p, g in zip(preds, gts):
            p = p[p[:, 1] >= t]
            m = match(p, g)
            tp += m.sum(); fp += (~m).sum(); ngt += len(g)
            cnt_ok += all((p[:, 0] == c).sum() == (g[:, 0] == c).sum() for c in range(NC))
        P, R = tp / max(tp + fp, 1), tp / max(ngt, 1)
        F = 2 * P * R / max(P + R, 1e-9)
        rows.append((t, P, R, F, cnt_ok / len(gts)))
        print(f"{t:5.2f} {P:9.4f} {R:7.4f} {F:6.4f} {cnt_ok / len(gts):13.3f} {fp / len(gts):11.3f}")
    best = max(rows, key=lambda r: r[4])
    print(f"-> conf {best[0]:.2f} cho tỉ lệ ảnh đếm đúng mọi lớp cao nhất ({best[4]:.3f}); "
          f"precision {best[1]:.4f}, recall {best[2]:.4f}")


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=str(MODELS / "det_model.tflite"))
    ap.add_argument("--weights", default=str(ROOT / "pipeline/weights/yolo_kd_student.pt"),
                    help="model float để so (bỏ qua: --weights '')")
    ap.add_argument("--images", default=str(ROOT / "outputs/yolo_tiny/ds_split/images/val"))
    ap.add_argument("--n", type=int, default=0, help="0 = tất cả")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--agnostic", type=int, default=0)
    ap.add_argument("--dist-scale", type=float, default=1.0,
                    help="hệ số tầm với P4 lúc train (tools/yolo_reach.py), vd 1.5; phải khớp YOLO_DIST_SCALE_LAST")
    args = ap.parse_args()
    ds = args.dist_scale

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    imgs = sorted(Path(args.images).glob("*.png")) + sorted(Path(args.images).glob("*.jpg"))
    imgs = imgs[:args.n] if args.n else imgs
    gts = [load_gt(p) for p in imgs]
    print(f"{len(imgs)} ảnh, {sum(len(g) for g in gts)} vật có nhãn")

    it = tf.lite.Interpreter(model_path=args.tflite)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    s_in, z_in = inp["quantization"]
    outs = it.get_output_details()
    reg_max = max(o["shape"][3] for o in outs) // 4
    levels = sorted({o["shape"][1] for o in outs}, reverse=True)       # H giảm dần: P2, P3, P4
    by = {(o["shape"][1], o["shape"][3]): o for o in outs}
    order = [(by[(h, 4 * reg_max)], by[(h, NC)]) for h in levels]

    build = FW / "build_host"
    build.mkdir(exist_ok=True)
    exe = build / ("yolo_host.exe" if os.name == "nt" else "yolo_host")
    subprocess.check_call(["gcc", "-O2", f"-DYOLO_DIST_SCALE_LAST={ds}f", "-I", str(FW / "main"),
                           str(FW / "test/yolo_host.c"),
                           str(FW / "main/logic.c"), "-lm", "-o", str(exe)])
    dump = build / "dump.bin"

    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(args.weights).model.float().eval().fuse()

    pa, pb = [], []
    deq = lambda t, o: (t.astype(np.float32) - o["quantization"][1]) * o["quantization"][0]
    with open(dump, "wb") as f:
        f.write(struct.pack("<5i", len(imgs), len(levels), reg_max, W_IN, H_IN))
        for k, p in enumerate(imgs):
            im = load_rgb(p)
            x = im.astype(np.float32)[None] / 255.0
            it.set_tensor(inp["index"], np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8))
            it.invoke()
            q = [(it.get_tensor(b["index"])[0], it.get_tensor(c["index"])[0]) for b, c in order]
            pb.append(decode_py([(deq(qb, b), deq(qc, c)) for (qb, qc), (b, c) in zip(q, order)], reg_max,
                                iou=args.iou, agnostic=bool(args.agnostic), dist_scale_last=ds))
            for (qb, qc), (b, c) in zip(q, order):
                for t, o in ((qb, b), (qc, c)):
                    f.write(struct.pack("<3ifi", *t.shape, o["quantization"][0], o["quantization"][1]))
                    f.write(np.ascontiguousarray(t).tobytes())
            if model is not None:
                with torch.no_grad():
                    fo = torch_head_outputs(model, torch.from_numpy(x).permute(0, 3, 1, 2).contiguous())
                pa.append(decode_py([(fo[2 * i][0], fo[2 * i + 1][0]) for i in range(len(levels))], reg_max,
                                    iou=args.iou, agnostic=bool(args.agnostic), dist_scale_last=ds))
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(imgs)}", flush=True)

    out = subprocess.run([str(exe), str(dump), "0.01", str(args.iou), str(args.agnostic)],
                         capture_output=True, text=True, check=True).stdout
    pc = [[] for _ in imgs]
    for line in out.splitlines():
        v = line.split()
        pc[int(v[0])].append([float(t) for t in v[1:]])
    pc = [np.array(r, np.float32).reshape(-1, 6) for r in pc]

    print("\nmAP@0.5 (cùng một cách tính, chỉ để so tương đối):")
    if pa:
        print(f"  A. PyTorch float + decode Python : {map50(pa, gts)[0]:.4f}")
    mb = map50(pb, gts)
    print(f"  B. TFLite INT8   + decode Python : {mb[0]:.4f}")
    mc = map50(pc, gts)
    print(f"  C. TFLite INT8   + decode C      : {mc[0]:.4f}   <- con số firmware sẽ đạt")
    print("     theo lớp: " + "  ".join(f"{n} {a:.3f}" for n, a in zip(NAMES, mc[1])))
    nb = sum(len(r[r[:, 1] >= 0.25]) for r in pb)
    nc_ = sum(len(r[r[:, 1] >= 0.25]) for r in pc)
    print(f"  số box conf>=0.25: Python {nb}, C {nc_}")
    if abs(mb[0] - mc[0]) > 0.003:
        print("  CẢNH BÁO: B và C lệch nhau > 0.003 -> kiểm tra decode_yolo trong main/logic.c")
    threshold_table(pc, gts, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])


if __name__ == "__main__":
    main()
