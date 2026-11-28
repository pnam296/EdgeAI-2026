"""Chỉnh ngưỡng conf / NMS theo ĐỘ CHÍNH XÁC CÂU TRẢ LỜI (luật chung kết), không theo mAP.

Chung kết chấm câu trả lời: COUNT+EXIST 35%, LIST 25%, LOCATION 30% (trái/giữa/phải theo tâm box),
OUT_OF_SCOPE 10% (chỉ phụ thuộc TextCNN, không tính ở đây). Script giả lập các câu hỏi đó trên ảnh có nhãn:
  - COUNT / EXIST : mỗi ảnh hỏi về các lớp CÓ trong ảnh và các lớp KHÔNG có (trộn theo --absent-weight)
  - LIST          : đúng khi số lượng MỌI lớp khớp nhãn
  - LOCATION      : hỏi lớp có trong ảnh mà mọi vật lớp đó cùng 1 vùng (câu hỏi không mơ hồ)
rồi tìm: NMS IoU, NMS chung lớp (agn), conf chung -> conf riêng từng lớp -> ngưỡng dự phòng LOCATION.

Hậu xử lý giống hệt firmware (main/logic.c): decode DFL (+ hệ số tầm với P4), chặn 512 ứng viên, NMS tham lam
(IoU > ngưỡng thì bỏ), giữ tối đa 64 box, rồi lọc theo ngưỡng từng lớp (answer_params_t). Với TFLite INT8
là đúng model chạy trên ESP.

    python tools/tune_answer.py                                   # model ESP hiện tại, BTC val + Roboflow val
    python tools/tune_answer.py --weights ../kaggle_runs/cam1/out/yolo_cam_best.pt --dist-scale 1.5
    python tools/tune_answer.py --yolo ../data/cam_test             # thêm tập ảnh camera thật (images/ + labels/)

In ra đoạn #define để dán vào main/model_config.h, lưu chi tiết vào outputs/tune_answer.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_det_yolo import H_IN, W_IN, load_rgb  # noqa: E402
from tfl_utils import FW, MODELS  # noqa: E402

NC = 7
NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
W_CE, W_LIST, W_LOC = 0.35, 0.25, 0.30
MAX_CAND, MAX_DETS = 512, 64          # = MAX_CAND trong logic.c, MAX_DETS trong main.c
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
REPO = FW.parent


# ------------------------------------------------------------------ dữ liệu
def read_gt(lbl: Path) -> np.ndarray:
    """[[cls, x1, y1, x2, y2]] chuẩn hóa 0..1"""
    if not lbl.exists():
        return np.zeros((0, 5), np.float32)
    rows = [l.split() for l in lbl.read_text().splitlines() if len(l.split()) == 5]
    if not rows:
        return np.zeros((0, 5), np.float32)
    c, x, y, w, h = np.array(rows, np.float32).T
    return np.stack([c, x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def btc_val(train_dir: Path, ratio=0.2, seed=42):
    """1.200 ảnh val giống pipeline (SPLIT_SEED 42) -> đo được trên ảnh model chưa thấy khi train."""
    imgs = sorted(p for p in (train_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    random.Random(seed).shuffle(imgs)
    return [(p, train_dir / "labels" / (p.stem + ".txt")) for p in imgs[:max(1, int(len(imgs) * ratio))]]


def rf_val(rf_sim: Path):
    """5% Roboflow giữ riêng giống kaggle_runs/cam1/train_cam.py (chỉ bản mô phỏng camera)."""
    imgs = sorted(p for p in (rf_sim / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    bases = sorted({p.stem.rsplit("_sim", 1)[0] for p in imgs})
    random.Random(1).shuffle(bases)
    val = set(bases[:max(1, len(bases) // 20)])
    return [(p, rf_sim / "labels" / (p.stem + ".txt")) for p in imgs
            if "_sim" in p.stem and p.stem.rsplit("_sim", 1)[0] in val]


def yolo_dir(d: Path):
    return [(p, d / "labels" / (p.stem + ".txt")) for p in sorted((d / "images").iterdir())
            if p.suffix.lower() in IMG_EXT]


# ------------------------------------------------------------------ model -> logit đầu ra từng mức
class Model:
    def __init__(self, tflite: str | None, weights: str | None):
        self.kind = "tflite" if tflite else "pt"
        if tflite:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            self.it = tf.lite.Interpreter(model_path=tflite)
            self.it.allocate_tensors()
            self.inp = self.it.get_input_details()[0]
            outs = self.it.get_output_details()
            self.reg_max = max(o["shape"][3] for o in outs) // 4
            levels = sorted({o["shape"][1] for o in outs}, reverse=True)      # P2, P3, P4
            by = {(o["shape"][1], o["shape"][3]): o for o in outs}
            self.order = [(by[(h, 4 * self.reg_max)], by[(h, NC)]) for h in levels]
        else:
            from ultralytics import YOLO
            self.m = YOLO(weights).model.float().eval().fuse()
            self.reg_max = int(self.m.model[-1].reg_max)

    def outputs(self, rgbs: list[np.ndarray]):
        """-> mỗi ảnh: [(box HxWx4R, cls HxWxNC)] float, thứ tự P2, P3, P4"""
        if self.kind == "tflite":
            s, z = self.inp["quantization"]
            res = []
            for im in rgbs:
                x = np.clip(np.round(im.astype(np.float32)[None] / 255.0 / s + z), -128, 127).astype(np.int8)
                self.it.set_tensor(self.inp["index"], x)
                self.it.invoke()
                deq = lambda o: (self.it.get_tensor(o["index"])[0].astype(np.float32) - o["quantization"][1]) \
                    * o["quantization"][0]
                res.append([(deq(b), deq(c)) for b, c in self.order])
            return res
        import torch
        from export_det_yolo import torch_head_outputs
        x = torch.from_numpy(np.stack(rgbs).astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        fo = torch_head_outputs(self.m, x)
        return [[(fo[2 * l][i], fo[2 * l + 1][i]) for l in range(len(fo) // 2)] for i in range(len(rgbs))]


def candidates(outs, reg_max, floor, ds_last):
    """decode giống decode_yolo: lớp = argmax logit, score = sigmoid(max), chặn MAX_CAND box điểm cao nhất"""
    C, S, B = [], [], []
    for li, (b, c) in enumerate(outs):
        H, W = c.shape[:2]
        stride = H_IN / H
        sc = 1 / (1 + np.exp(-c.max(-1)))
        ys, xs = np.nonzero(sc >= floor)
        if not len(ys):
            continue
        d = b[ys, xs].reshape(-1, 4, reg_max)
        d = np.exp(d - d.max(-1, keepdims=True))
        d = (d / d.sum(-1, keepdims=True) * np.arange(reg_max)).sum(-1)
        if li == len(outs) - 1:
            d = d * ds_last
        ax, ay = xs + 0.5, ys + 0.5
        B.append(np.stack([(ax - d[:, 0]) * stride / W_IN, (ay - d[:, 1]) * stride / H_IN,
                           (ax + d[:, 2]) * stride / W_IN, (ay + d[:, 3]) * stride / H_IN], 1).clip(0, 1))
        S.append(sc[ys, xs])
        C.append(c[ys, xs].argmax(-1))
    if not S:
        return np.zeros(0, int), np.zeros(0, np.float32), np.zeros((0, 4), np.float32)
    C, S, B = np.concatenate(C), np.concatenate(S).astype(np.float32), np.concatenate(B).astype(np.float32)
    o = np.argsort(-S, kind="stable")[:MAX_CAND]
    return C[o], S[o], B[o]


def iou_matrix(b):
    lt = np.maximum(b[:, None, :2], b[None, :, :2])
    rb = np.minimum(b[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), -1)
    u = area[:, None] + area[None] - inter
    return np.where(u > 0, inter / np.maximum(u, 1e-12), 0)


def nms(cls, ious, thr, agnostic, cross_iou=1.0):
    """giống nms() trong logic.c: duyệt theo score giảm dần, bỏ box có IoU > thr với box đã giữ.
    Khi agnostic=0 và cross_iou < 1.0: chạy thêm tầng 2 cross-class NMS để bỏ 2 nhãn đè khít cùng 1 vật."""
    n = len(cls)
    dead = np.zeros(n, bool)
    keep = []
    same = None if agnostic else (cls[:, None] == cls[None])
    for i in range(n):
        if dead[i]:
            continue
        keep.append(i)
        if len(keep) >= MAX_DETS:
            break
        hit = ious[i] > thr
        dead |= hit if agnostic else (hit & same[i])
    keep = np.array(keep, int)
    if not agnostic and cross_iou < 1.0 and len(keep) > 1:
        sub = nms(cls[keep], ious[np.ix_(keep, keep)], cross_iou, 1, 1.0)
        keep = keep[sub]
    return keep


def region(cx):
    return np.where(cx < 1 / 3, 0, np.where(cx < 2 / 3, 1, 2))


# ------------------------------------------------------------------ chấm câu trả lời
class Evaluator:
    """Gom box sau NMS của mọi ảnh (1 cấu hình NMS) để chấm nhanh nhiều bộ ngưỡng."""

    def __init__(self, kept, gts, absent_w):
        self.n = len(gts)
        img, cls, sc, rg = [], [], [], []
        for k, (c, s, r) in enumerate(kept):
            img.append(np.full(len(c), k)), cls.append(c), sc.append(s), rg.append(r)
        self.img, self.cls = np.concatenate(img).astype(int), np.concatenate(cls).astype(int)
        self.sc, self.rg = np.concatenate(sc), np.concatenate(rg).astype(int)
        self.key = self.img * NC + self.cls          # đã theo (ảnh, score giảm dần)
        g = np.zeros((self.n, NC), int)
        loc = np.full((self.n, NC), -1)               # vùng của lớp nếu mọi vật lớp đó cùng vùng, -2 = mơ hồ
        for k, gt in enumerate(gts):
            for c in range(NC):
                gc = gt[gt[:, 0] == c]
                g[k, c] = len(gc)
                if len(gc):
                    r = set(region((gc[:, 1] + gc[:, 3]) / 2).tolist())
                    loc[k, c] = r.pop() if len(r) == 1 else -2
        self.g, self.loc, self.absent_w = g, loc, absent_w

    def best_region(self, mask):
        """vùng của box điểm cao nhất mỗi (ảnh, lớp) trong mask, -1 nếu không có"""
        out = np.full(self.n * NC, -1)
        k = self.key[mask]
        u, first = np.unique(k, return_index=True)
        out[u] = self.rg[mask][first]
        return out.reshape(self.n, NC)

    def score(self, thr, loc_floor=None):
        thr = np.asarray(thr, np.float32)
        m = self.sc >= thr[self.cls]
        p = np.bincount(self.key[m], minlength=self.n * NC).reshape(self.n, NC)
        g = self.g
        pres, absn = g > 0, g == 0
        cnt_ok, ex_ok = p == g, (p > 0) == (g > 0)
        ce_p = np.concatenate([cnt_ok[pres], ex_ok[pres]]).mean() if pres.any() else 1.0
        ce_a = np.concatenate([cnt_ok[absn], ex_ok[absn]]).mean() if absn.any() else 1.0
        ce = (1 - self.absent_w) * ce_p + self.absent_w * ce_a
        lst = (p == g).all(1).mean()
        q = self.loc >= 0
        pr = self.best_region(m)
        if loc_floor is not None:
            fb = self.best_region(self.sc >= loc_floor)
            pr = np.where(pr >= 0, pr, fb)
        loc_p = (pr[q] == self.loc[q]).mean() if q.any() else 1.0
        qa = self.g == 0   # hỏi vị trí vật KHÔNG có trong ảnh: đúng khi không chỉ ra vị trí nào ("Không thấy ...")
        loc_a = (pr[qa] < 0).mean() if qa.any() else 1.0
        loc = (1 - self.absent_w) * loc_p + self.absent_w * loc_a
        total = (W_CE * ce + W_LIST * lst + W_LOC * loc) / (W_CE + W_LIST + W_LOC)
        return {"total": float(total), "count_exist": float(ce), "ce_present": float(ce_p),
                "ce_absent": float(ce_a), "list": float(lst), "location": float(loc),
                "loc_present": float(loc_p), "loc_absent": float(loc_a),
                "n_loc_q": int(q.sum())}


def combined(evals, weights, thr, loc_floor=None):
    rs = [e.score(thr, loc_floor) for e in evals]
    keys = [k for k in rs[0] if k != "n_loc_q"]
    tot = sum(weights)
    return {k: sum(w * r[k] for w, r in zip(weights, rs)) / tot for k in keys}, rs


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", default=str(MODELS / "det_model.tflite"), help="model INT8 (mặc định: đang nạp ESP)")
    ap.add_argument("--weights", default="", help=".pt thay cho --tflite (float, nhanh để thử)")
    ap.add_argument("--dist-scale", type=float, default=1.0, help="tầm với P4 lúc train (yolo_reach), vd 1.5")
    ap.add_argument("--btc", default=str(REPO / "data" / "train"), help="'' = bỏ BTC val")
    ap.add_argument("--rf", default=str(REPO / "data" / "rf_sim"), help="'' = bỏ Roboflow val")
    ap.add_argument("--yolo", action="append", default=[], help="thêm tập images/+labels/ (vd ảnh camera thật)")
    ap.add_argument("--set-weights", default="", help="trọng số từng tập theo thứ tự, vd '1,1,3' (mặc định đều)")
    ap.add_argument("--absent-weight", type=float, default=0.3, help="tỉ lệ câu COUNT/EXIST hỏi lớp KHÔNG có")
    ap.add_argument("--floor", type=float, default=0.08, help="ngưỡng thấp nhất khi lấy ứng viên")
    ap.add_argument("--n", type=int, default=0, help="giới hạn số ảnh mỗi tập (0 = tất cả)")
    ap.add_argument("--out", default=str(REPO / "outputs" / "tune_answer.json"))
    ap.add_argument("--eval", default="", help="chỉ CHẤM 1 cấu hình, không dò: 'conf,iou,agn,loc' vd '0.45,0.45,1,0.10'")
    ap.add_argument("--eval-cls", default="", help="ghi đè ngưỡng riêng vài lớp khi --eval, vd 'bottle=0.3,mouse=0.35'")
    ap.add_argument("--cross-nms", type=float, default=0.60, help="tầng 2 NMS khác lớp khi agn=0 (1.0 = tắt; khớp YOLO_CROSS_NMS_IOU)")
    args = ap.parse_args()

    sets = []
    if args.btc:
        sets.append(("BTC val", btc_val(Path(args.btc))))
    if args.rf:
        sets.append(("Roboflow val (mô phỏng camera)", rf_val(Path(args.rf))))
    for d in args.yolo:
        sets.append((Path(d).name, yolo_dir(Path(d))))
    if args.n:
        sets = [(n, s[:args.n]) for n, s in sets]
    wts = [float(v) for v in args.set_weights.split(",")] if args.set_weights else [1.0] * len(sets)
    assert len(wts) == len(sets), "--set-weights phải có đủ số tập"

    model = Model(None if args.weights else args.tflite, args.weights or None)
    print(f"model: {args.weights or args.tflite} | reg_max {model.reg_max} | tầm với P4 x{args.dist_scale}")
    IOUS = np.round(np.arange(0.30, 0.801, 0.05), 2)
    combos = [(t, a) for a in (1, 0) for t in IOUS]
    kept = {c: [] for c in combos}     # combo -> [ [ (cls, score, region) mỗi ảnh ] mỗi tập ]
    gts_all = []
    for name, items in sets:
        print(f"{name}: {len(items)} ảnh", flush=True)
        for c in combos:
            kept[c].append([])
        gts = []
        for k in range(0, len(items), 32):
            chunk = items[k:k + 32]
            outs = model.outputs([load_rgb(p) for p, _ in chunk])
            for (p, lp), o in zip(chunk, outs):
                gts.append(read_gt(lp))
                cl, sc, bx = candidates(o, model.reg_max, args.floor, args.dist_scale)
                ious = iou_matrix(bx)
                rg = region((bx[:, 0] + bx[:, 2]) / 2)
                for c in combos:
                    i = nms(cl, ious, c[0], c[1], args.cross_nms)
                    kept[c][-1].append((cl[i], sc[i], rg[i]))
        gts_all.append(gts)

    evals = {c: [Evaluator(kept[c][s], gts_all[s], args.absent_weight) for s in range(len(sets))] for c in combos}
    CONFS = np.round(np.arange(0.10, 0.851, 0.025), 3)

    if args.eval:
        cf, iou_e, agn_e, lf = (float(v) for v in args.eval.split(","))
        c = min(combos, key=lambda x: (abs(x[0] - iou_e), x[1] != int(agn_e)))
        thr_e = [cf] * NC
        for kv in filter(None, args.eval_cls.split(",")):
            k, v = kv.split("=")
            thr_e[NAMES.index(k.strip())] = float(v)
        r, per = combined(evals[c], wts, thr_e, lf)
        print(f"\nCHẤM conf {cf} | NMS IoU {c[0]} agn {c[1]} | LOCATION dự phòng {lf}")
        for (name, items), q in zip(sets, per):
            print(f"   {name:34s} tổng {q['total']:.4f} | C+E {q['count_exist']:.4f} (có {q['ce_present']:.3f}, "
                  f"không {q['ce_absent']:.3f}) | LIST {q['list']:.4f} | LOC {q['location']:.4f} (có {q['loc_present']:.3f}, không {q['loc_absent']:.3f}; {q['n_loc_q']} câu)")
        return

    # 1) NMS + conf chung
    best = None
    print(f"\n{'NMS IoU':>7} {'agn':>3} {'conf':>5} {'TỔNG':>6} {'CNT+EX':>6} {'LIST':>6} {'LOC':>6}")
    for c in combos:
        rows = [(cf, combined(evals[c], wts, [cf] * NC)[0]) for cf in CONFS]
        cf, r = max(rows, key=lambda x: x[1]["total"])
        print(f"{c[0]:7.2f} {c[1]:3d} {cf:5.3f} {r['total']:6.3f} {r['count_exist']:6.3f} {r['list']:6.3f} "
              f"{r['location']:6.3f}")
        if best is None or r["total"] > best[2]["total"]:
            best = (c, cf, r)
    (iou_b, agn_b), conf_b, r_b = best
    ev = evals[(iou_b, agn_b)]
    print(f"\n-> NMS IoU {iou_b}, agn {agn_b}, conf chung {conf_b}: tổng {r_b['total']:.4f}")
    print(f"\nĐường conf (NMS IoU {iou_b}, agn {agn_b}):")
    print(f"{'conf':>5} {'TỔNG':>6} {'CNT+EX':>6} {'(có)':>6} {'(không)':>7} {'LIST':>6} {'LOC':>6}")
    for cf in CONFS[::2]:
        r = combined(ev, wts, [cf] * NC)[0]
        print(f"{cf:5.3f} {r['total']:6.3f} {r['count_exist']:6.3f} {r['ce_present']:6.3f} {r['ce_absent']:7.3f} "
              f"{r['list']:6.3f} {r['location']:6.3f}")

    # 2) conf riêng từng lớp (coordinate ascent, chỉ nhận khi lợi >= 0.0005 để bớt khớp quá)
    thr = np.full(NC, conf_b, np.float32)
    cur = combined(ev, wts, thr)[0]["total"]
    for _ in range(3):
        changed = False
        for k in range(NC):
            for cf in CONFS:
                t2 = thr.copy()
                t2[k] = cf
                s = combined(ev, wts, t2)[0]["total"]
                if s > cur + 5e-4:
                    thr, cur, changed = t2, s, True
        if not changed:
            break
    print(f"\nconf riêng từng lớp: tổng {cur:.4f} (conf chung {r_b['total']:.4f})")
    print("  " + "  ".join(f"{n} {t:.3f}" for n, t in zip(NAMES, thr)))

    # 3) dự phòng LOCATION: không có box >= conf lớp thì lấy box điểm cao nhất >= loc_floor
    loc_rows = [(lf, combined(ev, wts, thr, lf)[0]) for lf in np.round(np.arange(args.floor, thr.max() + 1e-6, 0.025), 3)]
    base_loc = combined(ev, wts, thr)[0]["location"]
    lf_b, r_lf = max(reversed(loc_rows), key=lambda x: x[1]["location"])   # hòa -> ngưỡng cao hơn (an toàn)
    print(f"\nLOCATION dự phòng: không dùng {base_loc:.4f} -> loc_floor {lf_b:.3f}: {r_lf['location']:.4f}")
    for lf, r in loc_rows[::2]:
        print(f"  loc_floor {lf:.3f}: LOC {r['location']:.4f}")

    final, per = combined(ev, wts, thr, lf_b)
    print(f"\n== Cấu hình chọn: tổng {final['total']:.4f} | COUNT+EXIST {final['count_exist']:.4f} | "
          f"LIST {final['list']:.4f} | LOCATION {final['location']:.4f}")
    for (name, items), r in zip(sets, per):
        print(f"   {name:34s} tổng {r['total']:.4f} | C+E {r['count_exist']:.4f} (có {r['ce_present']:.3f}, "
              f"không {r['ce_absent']:.3f}) | LIST {r['list']:.4f} | LOC {r['location']:.4f} ({r['n_loc_q']} câu)")
    snippet = (f"#define CONF_THRESH      {conf_b:.3f}f\n"
               f"#define NMS_IOU          {iou_b:.2f}f\n"
               f"#define NMS_AGNOSTIC     {agn_b}\n"
               f"#define CLASS_CONF_INIT  {{{', '.join(f'{t:.3f}f' for t in thr)}}}\n"
               f"#define LOC_FLOOR        {lf_b:.3f}f")
    print("\nDán vào main/model_config.h:\n" + snippet)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.weights or args.tflite, "dist_scale": args.dist_scale,
                               "sets": [n for n, _ in sets], "nms_iou": float(iou_b), "agnostic": int(agn_b),
                               "conf": float(conf_b), "class_conf": dict(zip(NAMES, map(float, thr))),
                               "loc_floor": float(lf_b), "final": final, "per_set": per, "snippet": snippet},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print("đã lưu", out)


if __name__ == "__main__":
    main()
