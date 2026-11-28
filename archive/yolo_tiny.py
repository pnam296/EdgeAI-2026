"""Detector YOLO (Ultralytics) train TỪ ĐẦU: thầy to -> distill xuống trò <= 200k tham số. Chạy trên máy (GPU).

Mô hình (yolo_cfg/):
  student_p2p4.yaml  TRÒ 191k tham số: YOLO11 tự thiết kế cho vật nhỏ, đầu ra P2/4, P3/8, P4/16 (bỏ P5),
                     giảm kích thước bằng SCDown để dồn tham số sang P2/P3, reg_max 4.
  teacher_p2p4.yaml  THẦY 2,58 triệu (KHÔNG nộp): cùng bố cục đầu ra (6.300 anchor ở 320x240) để distill từng ô.
Bỏ P5 -> stride lớn nhất 16 -> đầu vào giữ nguyên 320x240 (240 chia hết cho 16, không phải co ảnh).
Không dùng pretrained: model dựng từ yaml. Thầy cũng train từ đầu trên dữ liệu BTC (luật cho phép distill kiểu này).

Distill (TEACHER = best.pt của thầy): ngoài loss thường, trò học theo thầy ở MỌI anchor:
  - lớp: BCE(logit trò, sigmoid(logit thầy))
  - box: KL(phân bố DFL thầy || trò), trọng số = độ tự tin của thầy ở ô đó
  cả hai chia cho tổng độ tự tin của thầy (cùng thang với loss lớp của Ultralytics).

Dự đoán tự làm (không qua model.predict): đưa đúng ảnh 320x240 vào model rồi NMS -> đầu vào không bị
Ultralytics đệm lên 320x256.

Chạy (cmd/PowerShell trong D:\\edgeai\\edgeai):
  .\\Scripts\\python.exe yolo_tiny.py PREP_ONLY=True                       # tạo dữ liệu 1 lần
  .\\Scripts\\python.exe yolo_tiny.py RUN='teacher' MODEL_CFG='yolo_cfg/teacher_p2p4.yaml' TEACHER_MODE=True EPOCHS=200
  .\\Scripts\\python.exe yolo_tiny.py RUN='student' EPOCHS=300             # trò không distill (đối chứng)
  .\\Scripts\\python.exe yolo_tiny.py RUN='kd' TEACHER='outputs/yolo_tiny/teacher/weights/best.pt' EPOCHS=300
Ra (outputs/yolo_tiny/<RUN>/): weights/, yolo_float.pt, det_test.json, submission.csv (intent lấy từ REF_SUB).
mAP val in ra tính bằng ĐÚNG hàm của edgeai_float.py, cùng tập val (seed 42) -> so trực tiếp với 0,9722.
"""
import ast
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

# ------------------------------- CẤU HÌNH -------------------------------
RUN = "student"               # tên thư mục kết quả trong OUT
MODEL_CFG = "yolo_cfg/student_p2p4.yaml"
TEACHER_MODE = False          # True: đang train THẦY -> bỏ kiểm tra trần tham số, không ghi submission
TEACHER = ""                  # best.pt của thầy -> bật distill
KD_CLS, KD_BOX = 1.0, 1.0     # hệ số loss distill lớp / box
EPOCHS = 300
BATCH = 64
W_IN, H_IN = 320, 240         # đầu vào model = kích thước ảnh gốc
MAX_PARAMS = 200_000
TRAIN_ON_ALL = False          # True: gộp val vào train (mAP val khi đó là GIẢ), dùng last.pt để nộp
SPLIT_SEED = 42               # GIỮ 42: cùng tập val với edgeai_float.py
SEED = 0
WORKERS = 2                   # Windows: mỗi worker là 1 tiến trình riêng, tốn RAM -> để thấp
CACHE = False                 # False: đọc PNG mỗi lần (ít RAM); "ram" nhanh hơn nhưng tốn ~1,2 GB/tiến trình
DEVICE = 0
CLOSE_MOSAIC = 10
NMS_IOU = 0.7
MAX_DET = 100
PREP_ONLY = False             # True: chỉ tạo dữ liệu kiểu Ultralytics rồi thoát
REF_SUB = "kaggle_runs/final-ep400/output/submission.csv"   # lấy phần INTENT (accuracy 1,0) từ bài cũ
DATA_DIR = ""

for _a in sys.argv[1:]:       # ghi đè KEY=VALUE (vd EPOCHS=3 TRAIN_ON_ALL=True RUN='kd')
    _k, _v = _a.split("=", 1)
    assert _k in globals(), f"Không có cấu hình {_k}"
    try:
        globals()[_k] = ast.literal_eval(_v)
    except (ValueError, SyntaxError):
        globals()[_k] = _v

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


# ------------------------------- ĐƯỜNG DẪN -------------------------------
def find_data_dir():
    if DATA_DIR:
        return Path(DATA_DIR)
    for root in (Path("/kaggle/input"), Path("/content/data"), Path("data"), Path(".")):
        if root.exists():
            hits = sorted(root.rglob("classes.txt"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("Không tìm thấy classes.txt")


DATA = find_data_dir()
OUT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs") / "yolo_tiny").resolve()
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_IMG, TRAIN_LBL, TEST_IMG = DATA / "train" / "images", DATA / "train" / "labels", DATA / "test" / "images"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def split_train_val(val_ratio=0.2, seed=42):
    """Giống hệt edgeai_float.split_train_val -> cùng tập val."""
    imgs = list_images(TRAIN_IMG)
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    return imgs[n_val:], imgs[:n_val]


# ------------------------------- mAP@0.5 (chép từ edgeai_float.py) -------------------------------
def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    ix1, iy1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    ix2, iy2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def average_precision(tp, n_gt):
    if n_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, dtype=float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = ctp / n_gt, ctp / (ctp + cfp)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return ap / 101


def map50(preds, gts, num_classes, iou_thr=0.5):
    per_class = {}
    for c in range(num_classes):
        gt_c = {k: np.array([b for cc, b in v if cc == c], dtype=float).reshape(-1, 4) for k, v in gts.items()}
        n_gt = sum(len(v) for v in gt_c.values())
        if n_gt == 0:
            continue
        dets = [(k, p["score"], p["bbox"]) for k, ps in preds.items() if k in gts for p in ps if p["class_id"] == c]
        dets.sort(key=lambda d: -d[1])
        used = {k: np.zeros(len(v), bool) for k, v in gt_c.items()}
        tp = []
        for k, _, box in dets:
            g = gt_c.get(k)
            if g is None or len(g) == 0:
                tp.append(0)
                continue
            ious = iou(np.asarray(box, float), g)
            ious[used[k]] = -1
            j = int(ious.argmax())
            if ious[j] >= iou_thr:
                used[k][j] = True
                tp.append(1)
            else:
                tp.append(0)
        per_class[c] = average_precision(tp, n_gt)
    return float(np.mean(list(per_class.values()))), per_class


def read_gt(img_path, W0, H0):
    out = []
    f = TRAIN_LBL / (img_path.stem + ".txt")
    for line in f.read_text().splitlines() if f.exists() else []:
        v = line.split()
        if len(v) == 5:
            c, x, y, w, h = int(float(v[0])), *map(float, v[1:])
            out.append((c, [(x - w / 2) * W0, (y - h / 2) * H0, (x + w / 2) * W0, (y + h / 2) * H0]))
    return out


# ------------------------------- DỮ LIỆU KIỂU ULTRALYTICS -------------------------------
def build_dataset(classes, train_paths, val_paths):
    """images/{train,val} + labels/{train,val}. Dùng chung giữa các run (thầy, trò chạy song song) -> tạo 1 lần."""
    ds = OUT / f"ds_{'all' if TRAIN_ON_ALL else 'split'}"
    if (ds / "data.yaml").exists():
        return ds
    if ds.exists():
        shutil.rmtree(ds)
    for split, paths in (("train", train_paths), ("val", val_paths)):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for p in paths:
            shutil.copy2(p, ds / "images" / split / p.name)
            src = TRAIN_LBL / (p.stem + ".txt")
            (ds / "labels" / split / (p.stem + ".txt")).write_text(src.read_text() if src.exists() else "")
    cfg = {"path": str(ds), "train": "images/train", "val": "images/val", "names": dict(enumerate(classes))}
    (ds / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return ds


# ------------------------------- DISTILL -------------------------------
class KDLoss(v8DetectionLoss):
    """Loss thường của Ultralytics + học theo đầu ra của thầy ở từng anchor (cùng bố cục 6.300 anchor)."""

    def __init__(self, model, teacher):
        super().__init__(model)
        self.teacher = teacher

    def loss(self, preds, batch):
        total, items = super().loss(preds, batch)
        with torch.no_grad():
            tp = self.parse_output(self.teacher(batch["img"]))
        B = preds["scores"].shape[0]
        s_cls, t_cls = preds["scores"].float(), tp["scores"].float()
        t_p = t_cls.sigmoid()
        norm = t_p.sum().clamp(min=1)
        kd_cls = F.binary_cross_entropy_with_logits(s_cls, t_p, reduction="sum") / norm
        R = self.reg_max
        s_box = preds["boxes"].float().view(B, 4, R, -1)
        t_box = tp["boxes"].float().view(B, 4, R, -1)
        kl = (F.softmax(t_box, 2) * (F.log_softmax(t_box, 2) - F.log_softmax(s_box, 2))).sum(2)   # B, 4, A
        kd_box = (kl.sum(1) * t_p.max(1).values).sum() / norm
        return total + (KD_CLS * kd_cls + KD_BOX * kd_box) * B, items


def attach_kd(trainer):
    teacher = YOLO(TEACHER).model.to(trainer.device).float().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_out = teacher(torch.zeros(1, 3, H_IN, W_IN, device=trainer.device))
    s_out = trainer.model.eval()(torch.zeros(1, 3, H_IN, W_IN, device=trainer.device))
    trainer.model.train()
    assert t_out[1]["boxes"].shape == s_out[1]["boxes"].shape, "Thầy và trò khác bố cục đầu ra"
    trainer.model.criterion = KDLoss(trainer.model, teacher)
    print(f"Distill từ {TEACHER} ({sum(p.numel() for p in teacher.parameters()):,} tham số), "
          f"KD_CLS {KD_CLS}, KD_BOX {KD_BOX}", flush=True)


# ------------------------------- DỰ ĐOÁN -------------------------------
@torch.no_grad()
def predict(weights, paths, iou_nms=NMS_IOU):
    """Ảnh gốc 320x240 đưa THẲNG vào model (không đệm), NMS kiểu Ultralytics -> box pixel ảnh gốc."""
    model = YOLO(str(weights)).model.to(DEVICE).float().eval()
    out = {}
    for k in range(0, len(paths), 64):
        chunk = paths[k:k + 64]
        imgs = []
        for p in chunk:
            im = cv2.imread(str(p))
            assert im.shape[:2] == (H_IN, W_IN), (p, im.shape)
            imgs.append(im[:, :, ::-1].transpose(2, 0, 1))   # BGR -> RGB, HWC -> CHW
        x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs))).to(DEVICE).float() / 255
        y = model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        dets = non_max_suppression(y, conf_thres=0.001, iou_thres=iou_nms, max_det=MAX_DET)
        for p, d in zip(chunk, dets):
            d = d.cpu().numpy()
            out[p.stem] = [{"class_id": int(c), "score": round(float(s), 4),
                            "bbox": [round(float(min(max(x1, 0), W_IN)), 1), round(float(min(max(y1, 0), H_IN)), 1),
                                     round(float(min(max(x2, 0), W_IN)), 1), round(float(min(max(y2, 0), H_IN)), 1)]}
                           for x1, y1, x2, y2, s, c in d[:, :6]]
    return out


def main():
    t0 = time.time()
    classes = [l.strip() for l in (DATA / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    C = len(classes)
    train_p, val_p = split_train_val(seed=SPLIT_SEED)
    if TRAIN_ON_ALL:
        train_p = train_p + val_p
    ds = build_dataset(classes, train_p, val_p)
    print(f"Ultralytics {ultralytics.__version__} | {RUN} | {MODEL_CFG} | train {len(train_p)} | val {len(val_p)} "
          f"| đầu vào {W_IN}x{H_IN} | TRAIN_ON_ALL {TRAIN_ON_ALL} | dữ liệu {ds} ({time.time() - t0:.0f}s)", flush=True)
    if PREP_ONLY:
        return

    model = YOLO(MODEL_CFG)                   # dựng TỪ YAML -> trọng số ngẫu nhiên, không pretrained
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"Số tham số: {n_params:,} (giới hạn {MAX_PARAMS:,}){' - THẦY, không nộp' if TEACHER_MODE else ''}", flush=True)
    assert TEACHER_MODE or n_params <= MAX_PARAMS, "VƯỢT trần tham số"
    if TEACHER:
        model.add_callback("on_train_start", attach_kd)
    model.train(data=str(ds / "data.yaml"), epochs=EPOCHS, imgsz=W_IN, batch=BATCH, workers=WORKERS,
                device=DEVICE, cache=CACHE, pretrained=False, seed=SEED, deterministic=False,
                project=str(OUT), name=RUN, exist_ok=True, plots=False, patience=0,
                close_mosaic=CLOSE_MOSAIC, verbose=False)
    wdir = Path(model.trainer.save_dir) / "weights"
    print(f"Train xong sau {(time.time() - t0) / 60:.1f} phút | trọng số: {wdir}", flush=True)

    # Đánh giá trên val bằng ĐÚNG hàm mAP của edgeai_float.py (so được với val 0,9722 của lr3ep240)
    gts = {p.stem: read_gt(p, W_IN, H_IN) for p in val_p}
    scores = {}
    for wname in ("best.pt", "last.pt"):
        for nms in sorted({NMS_IOU, 0.5, 0.6}):
            mp, per = map50(predict(wdir / wname, val_p, iou_nms=nms), gts, C)
            scores[(wname, nms)] = mp
            print(f"mAP50 val {wname} NMS {nms}: {mp:.4f} | " + " ".join(f"{classes[c]} {a:.3f}" for c, a in per.items()),
                  flush=True)
    if TRAIN_ON_ALL:
        print("  (TRAIN_ON_ALL: mAP val là GIẢ, val nằm trong train)")
    if TEACHER_MODE:
        print(f"THẦY xong. Distill: TEACHER='{(wdir / 'best.pt').as_posix()}'")
        return
    wname = "last.pt" if TRAIN_ON_ALL else max(("best.pt", "last.pt"), key=lambda w: scores[(w, NMS_IOU)])
    print(f"Dùng {wname}, NMS {NMS_IOU} để dự đoán test", flush=True)
    final = YOLO(str(wdir / wname)).model.float()
    n_final = sum(p.numel() for p in final.parameters())
    torch.save({k: v.float() for k, v in final.state_dict().items()}, OUT / RUN / "yolo_float.pt")
    kb = (OUT / RUN / "yolo_float.pt").stat().st_size / 1024
    print(f"Model nộp: {n_final:,} tham số {'OK' if n_final <= MAX_PARAMS else 'VƯỢT'} | trọng số float32 {kb:.0f} KB "
          f"(giới hạn 1536 KB) {'OK' if kb <= 1536 else 'VƯỢT'}")

    test_p = list_images(TEST_IMG)
    det = predict(wdir / wname, test_p)
    (OUT / RUN / "det_test.json").write_text(json.dumps(det, separators=(",", ":")), encoding="utf-8")
    print(f"Test: {len(det)} ảnh, {sum(map(len, det.values()))} box")

    ref = Path(REF_SUB)
    if not ref.exists():
        print(f"Không có {ref} -> chỉ ghi det_test.json, chưa ghép submission")
        return
    rows = []
    with open(ref, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["task_type"] == "DETECTION":
                r["prediction"] = json.dumps(det[r["sample_id"]], separators=(",", ":"))
            rows.append((r["task_type"], r["sample_id"], r["prediction"]))
    with open(OUT / RUN / "submission.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_type", "sample_id", "prediction"])
        w.writerows(rows)
    print(f"Ghi {OUT / RUN / 'submission.csv'}: {len(rows)} dòng, intent lấy từ {ref}")


if __name__ == "__main__":
    main()
