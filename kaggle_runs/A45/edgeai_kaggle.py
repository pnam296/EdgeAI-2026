# =====================================================================================
#  EDGE AI - VÒNG LOẠI: TẤT CẢ TRONG 1 FILE ĐỂ CHẠY TRÊN KAGGLE (GPU)
#  Gộp: step5 (detector tí hon -> INT8) + step2 (intent TextCNN) + step4 (submission.csv)
#
#  Cách dùng trên Kaggle:
#    1. Tải thư mục data/ lên làm Kaggle Dataset (giữ nguyên cấu trúc:
#         classes.txt, sample_submission.csv, intent_train.csv, intent_test.csv,
#         train/images, train/labels, test/images)
#    2. Notebook mới -> Add Input -> chọn dataset đó
#    3. Settings -> Accelerator -> "GPU T4 x2" (nên chọn T4, P100 quá cũ với TF mới)
#    4. Dán toàn bộ file này vào 1 ô code -> Run
#    5. Kết quả nằm trong /kaggle/working (tab Output): submission.csv, *.tflite, ...
# =====================================================================================
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import json
import math
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# ------------------------------- CẤU HÌNH -------------------------------
SEED = 42
# Detector
INPUT_W, INPUT_H = 320, 240   # mỗi chiều <= 160 và chia hết cho 16
ALPHA = 0.45                  # 0.25 -> ~79k tham số; 0.35 -> ~130k tham số
NECK_CH = 64
WH_LOG = False                # True: học log(w), log(h)
# ---- RUN A: 4 thay đổi, mỗi cái một cờ (đặt False = quay về như mốc 3-a35-160) ----
SPLIT_BOX = True              # (1) tách output box thành 2 tensor: wh (2C) và offset tâm (2C) -> mỗi cái 1 scale INT8
WH_SQRT = True                #     wh học sqrt(w/ảnh), sqrt(h/ảnh) (0..1) thay vì số ô (0..40) -> lượng tử mịn hơn
CALIB_N = 500                 # (2) số ảnh hiệu chỉnh INT8 (cũ: 100 ảnh đầu)
GIOU_WEIGHT = 2.0             # (3) thêm loss GIoU cho box (0 = tắt)
EMA_DECAY = 0.999             # (4) trung bình trượt trọng số; đánh giá + lưu + xuất bằng trọng số EMA (0 = tắt)
EPOCHS = 160                  # 6.000 ảnh -> mỗi epoch nhiều bước hơn Warm-up; xem ETA in ra rồi chỉnh
BATCH = 16
LR = 2e-3
EVAL_EVERY = 5
# Dừng sớm (PATIENCE > 0): bỏ cosine, giữ LR cố định, val mAP không tăng PLATEAU epoch -> LR x0.5;
# không tăng PATIENCE epoch (hoặc LR đã chạm sàn) -> dừng. EPOCHS khi đó chỉ là trần tối đa.
PATIENCE = 0                  # 0 = tắt (dùng cosine theo EPOCHS như cũ)
PLATEAU = 15
MIN_DELTA = 0.002             # tăng ít hơn mức này coi như không tiến bộ
MIN_LR_FRAC = 0.02            # sàn LR = LR * MIN_LR_FRAC
TRAIN_ON_ALL = False          # True: gộp val vào train khi đã chốt cấu hình (mAP val mất ý nghĩa)
WORKERS = 4                   # số luồng chuẩn bị dữ liệu song song (Kaggle có 4 nhân CPU)
# Intent
RUN_INTENT = True
INTENT_EPOCHS, INTENT_BATCH, MAX_LEN = 20, 64, 24
DICT_AUGMENT = True           # sinh câu train bằng từ điển tên vật (CHỈ khi train; dự đoán không dùng từ điển)
# Nhãn detection. "auto" = tự nhận dạng và DỪNG nếu không chắc chắn.
# Các giá trị khác: "yolo" (cls cx cy w h, 0..1) | "xyxy_px" (cls x1 y1 x2 y2, pixel)
#   | "xyxy_norm" (cls x1 y1 x2 y2, 0..1) | "xyxy_px_last" (x1 y1 x2 y2 cls, pixel) | "xyxy_norm_last"
LABEL_FORMAT = "auto"
# CHẠY THỬ: True = 300 ảnh train, 2 epoch -> kiểm tra toàn bộ pipeline trong vài phút
# (nhãn, train, INT8, submission). Chạy thử trước, ổn rồi mới đặt False để chạy thật.
QUICK_TEST = False

# ------------------------------- ĐƯỜNG DẪN -------------------------------
def find_data_dir():
    """Tìm thư mục chứa classes.txt trong /kaggle/input (hoặc ./data khi chạy ở máy)."""
    for root in (Path("/kaggle/input"), Path("data"), Path(".")):
        if root.exists():
            hits = sorted(root.rglob("classes.txt"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("Không tìm thấy classes.txt - đã Add Input dataset chưa?")


DATA = find_data_dir()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)
TRAIN_IMG, TRAIN_LBL, TEST_IMG = DATA / "train" / "images", DATA / "train" / "labels", DATA / "test" / "images"
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
print(f"Dữ liệu: {DATA}\nKết quả: {OUT}")

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print(f"TensorFlow {tf.__version__} | GPU: {[g.name for g in gpus] or 'KHÔNG CÓ (đang chạy CPU!)'}")
tf.keras.utils.set_random_seed(SEED)

assert INPUT_W % 16 == 0 and INPUT_H % 16 == 0 and INPUT_W <= 320 and INPUT_H <= 240
assert not (WH_LOG and WH_SQRT), "chọn một trong WH_LOG / WH_SQRT"
STRIDE = 8
GRID_W, GRID_H = INPUT_W // STRIDE, INPUT_H // STRIDE
CACHE_W = 640


# =====================================================================================
#  HÀM DÙNG CHUNG (common.py)
# =====================================================================================
def load_classes():
    lines = (DATA / "classes.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def read_label_rows(path):
    """Đọc file nhãn thành list các dòng số (tách theo khoảng trắng hoặc dấu phẩy; bỏ dòng tiêu đề)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue  # dòng tiêu đề / chữ
    return [r for r in rows if r]


def to_norm_xyxy(row, fmt, W0, H0):
    """Một dòng nhãn -> (cls, x1, y1, x2, y2) chuẩn hóa 0..1 theo định dạng fmt."""
    if fmt.endswith("_last"):
        row = [row[4]] + row[:4]
    c, a, b, d, e = row
    if fmt == "yolo":
        return c, a - d / 2, b - e / 2, a + d / 2, b + e / 2
    if fmt.startswith("xyxy_px"):
        return c, a / W0, b / H0, d / W0, e / H0
    return c, a, b, d, e  # xyxy_norm


def row_valid(row, fmt, W0, H0, n_classes, tol=0.02):
    """Dòng nhãn có hợp lệ với định dạng fmt không: đủ 5 cột, lớp là số nguyên 0..C-1, box nằm trong ảnh, x2 > x1."""
    if len(row) != 5:
        return False
    c, x1, y1, x2, y2 = to_norm_xyxy(row, fmt, W0, H0)
    if c != int(c) or not 0 <= c < n_classes:
        return False
    if fmt == "yolo" and not all(0 <= v <= 1 + tol for v in row[1:]):
        return False
    return (-tol <= x1 < x2 <= 1 + tol) and (-tol <= y1 < y2 <= 1 + tol)


def detect_label_format(img_paths, n_classes, n_check=300):
    """Thử mọi định dạng trên n_check ảnh đầu; chỉ chấp nhận khi ĐÚNG MỘT định dạng hợp lệ ở mọi dòng."""
    fmts = ["yolo", "xyxy_px", "xyxy_norm", "xyxy_px_last", "xyxy_norm_last"]
    ok = {f: 0 for f in fmts}
    total = 0
    px = {"first": False, "last": False}  # có tọa độ > 1.5 không, nếu lớp ở cột ĐẦU / cột CUỐI
    for p in img_paths[:n_check]:
        H0, W0 = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]
        for r in read_label_rows(TRAIN_LBL / (p.stem + ".txt")):
            total += 1
            px["first"] |= len(r) >= 5 and max(r[1:5]) > 1.5   # KHÔNG tính cột lớp (giá trị tới C-1)
            px["last"] |= len(r) >= 5 and max(r[0:4]) > 1.5
            for f in fmts:
                ok[f] += row_valid(r, f, W0, H0, n_classes)
    print(f"Kiểm tra định dạng nhãn trên {min(n_check, len(img_paths))} ảnh, {total} dòng:")
    for f in fmts:
        print(f"  {f:<15} hợp lệ {ok[f]}/{total}")
    # Có tọa độ > 1.5 -> nhãn PIXEL; toàn bộ tọa độ <= 1 -> CHUẨN HÓA (box rộng 0.3 "pixel" là vô lý)
    allowed = [f for f in fmts if ("_px" in f) == px["last" if f.endswith("_last") else "first"]]
    full = [f for f in allowed if total and ok[f] == total]
    print(f"  -> định dạng còn hợp lý sau khi xét pixel/chuẩn hóa: {allowed}")
    any_px = px["first"] or px["last"]
    if total == 0:
        raise RuntimeError(f"Không đọc được dòng nhãn nào trong {TRAIN_LBL}. Kiểm tra thư mục và đuôi file nhãn.")
    if len(full) == 1:
        return full[0]
    if not full:
        raise RuntimeError("Không định dạng nào khớp 100% -> in 3 file nhãn ở trên và gửi lại để sửa code.")
    if not any_px and set(full) >= {"yolo", "xyxy_norm"}:
        raise RuntimeError(f"Nhiều định dạng cùng hợp lệ {full}: nhãn chuẩn hóa 0..1 nhưng không phân biệt được "
                           "YOLO (cx cy w h) với (x1 y1 x2 y2). Xem ảnh preview rồi đặt LABEL_FORMAT bằng tay.")
    raise RuntimeError(f"Nhiều định dạng cùng hợp lệ {full} -> đặt LABEL_FORMAT bằng tay.")


def split_train_val(val_ratio=0.2, seed=42):
    imgs = list_images(TRAIN_IMG)
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    return imgs[n_val:], imgs[:n_val]


# =====================================================================================
#  mAP@0.5 (metrics.py)
# =====================================================================================
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


def map50(preds, gts, num_classes, iou_thr=0.5, return_per_class=False):
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
    m = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return (m, per_class) if return_per_class else m


# =====================================================================================
#  DỮ LIỆU DETECTOR (det_tiny_data.py)
# =====================================================================================
def load_image(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    if W0 > CACHE_W:
        img = cv2.resize(img, (CACHE_W, round(H0 * CACHE_W / W0)), interpolation=cv2.INTER_AREA)
    return img, (W0, H0)


def load_samples(img_paths, label_dir, fmt=None, n_classes=None):
    """Mỗi mẫu: dict(id, img, orig_size, boxes) với boxes = N x 5 [cls, x1, y1, x2, y2] chuẩn hóa 0..1."""
    samples, bad = [], 0
    for p in img_paths:
        img, (W0, H0) = load_image(p)
        rows = []
        if label_dir is not None:
            for r in read_label_rows(label_dir / (p.stem + ".txt")):
                if row_valid(r, fmt, W0, H0, n_classes):
                    rows.append(list(to_norm_xyxy(r, fmt, W0, H0)))
                else:
                    bad += 1
        boxes = np.clip(np.array(rows, dtype=np.float32).reshape(-1, 5), [0, 0, 0, 0, 0], [999, 1, 1, 1, 1])
        samples.append({"id": p.stem, "img": img, "orig_size": (W0, H0), "boxes": boxes})
    if bad:
        print(f"  CẢNH BÁO: bỏ {bad} dòng nhãn không hợp lệ")
    return samples


def preprocess(img_rgb):
    x = cv2.resize(img_rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return x / 127.5 - 1.0


def augment(img, boxes, rng):
    H, W = img.shape[:2]
    boxes = boxes.copy()
    if rng.random() < 0.7:
        s = rng.uniform(0.55, 1.0)
        cw, ch = s, min(1.0, s * rng.uniform(0.85, 1.15))
        x0, y0 = rng.uniform(0, 1 - cw), rng.uniform(0, 1 - ch)
        img = img[int(y0 * H): int((y0 + ch) * H), int(x0 * W): int((x0 + cw) * W)]
        if len(boxes):
            area0 = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
            b = boxes.copy()
            b[:, [1, 3]] = np.clip((b[:, [1, 3]] - x0) / cw, 0, 1)
            b[:, [2, 4]] = np.clip((b[:, [2, 4]] - y0) / ch, 0, 1)
            area1 = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2]) * cw * ch
            keep = (area1 > 0.4 * area0) & (b[:, 3] - b[:, 1] > 0.005) & (b[:, 4] - b[:, 2] > 0.005)
            boxes = b[keep]
    if rng.random() < 0.5:
        img = img[:, ::-1]
        if len(boxes):
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
    img = img.astype(np.float32)
    if rng.random() < 0.8:
        img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.6, 1.4)
    if rng.random() < 0.1:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), boxes


def encode_targets(boxes, num_classes):
    C = num_classes
    heat = np.zeros((GRID_H, GRID_W, C), np.float32)
    box = np.zeros((GRID_H, GRID_W, 5 * C), np.float32)
    ys, xs = np.mgrid[0:GRID_H, 0:GRID_W]
    order = np.argsort(-(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])) if len(boxes) else []
    for c, x1, y1, x2, y2 in boxes[order]:
        c = int(c)
        w, h = (x2 - x1) * GRID_W, (y2 - y1) * GRID_H
        cx, cy = (x1 + x2) / 2 * GRID_W, (y1 + y2) / 2 * GRID_H
        ix, iy = min(int(cx), GRID_W - 1), min(int(cy), GRID_H - 1)
        sx, sy = max(0.54 * w / 6, 0.4), max(0.54 * h / 6, 0.4)
        g = np.exp(-((xs - ix) ** 2) / (2 * sx * sx) - ((ys - iy) ** 2) / (2 * sy * sy))
        heat[:, :, c] = np.maximum(heat[:, :, c], g)
        heat[iy, ix, c] = 1.0
        if WH_LOG:
            w, h = np.log(max(w, 1e-3)), np.log(max(h, 1e-3))
        elif WH_SQRT:
            w, h = np.sqrt(max(x2 - x1, 0.0)), np.sqrt(max(y2 - y1, 0.0))
        box[iy, ix, 4 * c: 4 * c + 4] = [w, h, cx - ix, cy - iy]
        box[iy, ix, 4 * C + c] = 1.0
    return heat, box


def decode(heat, box, orig_size, max_det=100, min_score=0.001):
    W0, H0 = orig_size
    padded = np.pad(heat, ((1, 1), (1, 1), (0, 0)), constant_values=-1)
    local_max = np.max(np.stack([padded[dy:dy + GRID_H, dx:dx + GRID_W] for dy in range(3) for dx in range(3)]), axis=0)
    peaks = (heat >= local_max) & (heat >= min_score)
    iy, ix, c = np.nonzero(peaks)
    scores = heat[iy, ix, c]
    out = []
    for i in np.argsort(-scores)[:max_det]:
        w, h, dx, dy = box[iy[i], ix[i], 4 * c[i]: 4 * c[i] + 4]
        if WH_LOG:
            w, h = float(np.exp(min(w, 4.0))), float(np.exp(min(h, 4.0)))
        elif WH_SQRT:
            w, h = max(float(w), 0.0) ** 2 * GRID_W, max(float(h), 0.0) ** 2 * GRID_H
        w, h = max(float(w), 0.05), max(float(h), 0.05)
        cx, cy = (ix[i] + dx) / GRID_W, (iy[i] + dy) / GRID_H
        x1 = max(0.0, (cx - w / 2 / GRID_W) * W0)
        y1 = max(0.0, (cy - h / 2 / GRID_H) * H0)
        x2 = min(float(W0), (cx + w / 2 / GRID_W) * W0)
        y2 = min(float(H0), (cy + h / 2 / GRID_H) * H0)
        if x2 > x1 and y2 > y1:
            out.append({"class_id": int(c[i]), "score": round(float(scores[i]), 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return out


def gt_pixels(sample):
    W0, H0 = sample["orig_size"]
    return [(int(c), [x1 * W0, y1 * H0, x2 * W0, y2 * H0]) for c, x1, y1, x2, y2 in sample["boxes"]]


# =====================================================================================
#  MÔ HÌNH + LOSS + BỘ NẠP DỮ LIỆU (det_tiny_model.py)
# =====================================================================================
L = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="edgeai")
class ResizeNearest(L.Layer):
    """Phóng x2 kiểu nearest -> RESIZE_NEAREST_NEIGHBOR (có trong TFLite Micro)."""

    def __init__(self, size, **kw):
        super().__init__(**kw)
        self.size = tuple(size)

    def call(self, x):
        return tf.image.resize(x, self.size, method="nearest")

    def get_config(self):
        return {**super().get_config(), "size": self.size}


def _conv_bn_relu(x, ch, k=1, name=None):
    x = L.Conv2D(ch, k, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = L.BatchNormalization(name=f"{name}_bn")(x)
    return L.ReLU(6.0, name=f"{name}_relu")(x)


def _sep_conv(x, ch, name):
    x = L.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = L.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = L.ReLU(6.0, name=f"{name}_dw_relu")(x)
    return _conv_bn_relu(x, ch, 1, name=f"{name}_pw")


def build_model(num_classes):
    # Vòng loại: train từ đầu (weights=None)
    base = tf.keras.applications.MobileNetV2(input_shape=(INPUT_H, INPUT_W, 3), alpha=ALPHA,
                                             include_top=False, weights=None)
    c8 = base.get_layer("block_6_expand_relu").output
    c16 = base.get_layer("block_13_expand_relu").output
    p16 = _conv_bn_relu(c16, NECK_CH, name="lat16")
    p16 = ResizeNearest((INPUT_H // 8, INPUT_W // 8), name="up16")(p16)
    p8 = _conv_bn_relu(c8, NECK_CH, name="lat8")
    x = L.Add(name="fuse")([p8, p16])
    x = _sep_conv(x, NECK_CH, "neck1")
    x = _sep_conv(x, NECK_CH, "neck2")
    heat = L.Conv2D(num_classes, 1, activation="sigmoid", name="heatmap",
                    bias_initializer=tf.keras.initializers.Constant(-2.19))(x)
    if not SPLIT_BOX:
        box = L.Conv2D(4 * num_classes, 1, name="box")(x)
        m = tf.keras.Model(base.input, [heat, box], name="tiny_centernet")
        return m, m
    # Hai đầu riêng -> 2 tensor output riêng -> mỗi tensor có scale INT8 riêng
    # (gộp chung thì w,h 0..40 ô làm offset 0..1 bị làm tròn tới ~0,16 ô).
    wh = L.Conv2D(2 * num_classes, 1, name="box_wh")(x)
    off = L.Conv2D(2 * num_classes, 1, name="box_off")(x)
    # Bản train: xếp lại thành [w, h, dx, dy] cho từng lớp như cũ (Reshape/Concat chỉ có ở bản train, không xuất)
    g = (INPUT_H // STRIDE, INPUT_W // STRIDE)
    box = L.Concatenate(axis=-1, name="box_cat")([L.Reshape((*g, num_classes, 2))(wh),
                                                  L.Reshape((*g, num_classes, 2))(off)])
    box = L.Reshape((*g, 4 * num_classes), name="box")(box)
    train_m = tf.keras.Model(base.input, [heat, box], name="tiny_centernet")
    export_m = tf.keras.Model(base.input, [heat, wh, off], name="tiny_centernet_export")
    return train_m, export_m


def focal_loss(y_true, y_pred):
    p = tf.clip_by_value(y_pred, 1e-4, 1 - 1e-4)
    pos = tf.cast(tf.equal(y_true, 1.0), tf.float32)
    pos_loss = tf.math.log(p) * tf.square(1 - p) * pos
    neg_loss = tf.math.log(1 - p) * tf.square(p) * tf.pow(1 - y_true, 4) * (1 - pos)
    n_pos = tf.maximum(tf.reduce_sum(pos, axis=[1, 2, 3]), 1.0)
    return -tf.reduce_sum(pos_loss + neg_loss, axis=[1, 2, 3]) / n_pos


def make_box_loss(num_classes, off_weight=1.0):
    C = num_classes
    # sqrt(w/ảnh): sai 1 ô (~1/40 ảnh) với vật ~8 ô ≈ 0,028 đơn vị sqrt -> hệ số ~3,5 để cân bằng với mức cũ 0,1/ô
    wh_weight = 1.0 if WH_LOG else (4.0 if WH_SQRT else 0.1)

    def box_loss(y_true, y_pred):
        shape = tf.shape(y_pred)
        tgt = tf.reshape(y_true[..., :4 * C], [shape[0], shape[1], shape[2], C, 4])
        pred = tf.reshape(y_pred, [shape[0], shape[1], shape[2], C, 4])
        mask = y_true[..., 4 * C:][..., None]
        err = tf.abs(pred - tgt) * mask
        n = tf.maximum(tf.reduce_sum(mask, axis=[1, 2, 3, 4]), 1.0)
        wh = tf.reduce_sum(err[..., :2], axis=[1, 2, 3, 4])
        off = tf.reduce_sum(err[..., 2:], axis=[1, 2, 3, 4])
        loss = (wh_weight * wh + off_weight * off) / n
        if GIOU_WEIGHT > 0:
            def to_xyxy(t):   # đơn vị: ô lưới, gốc ở góc ô chứa tâm
                if WH_LOG:
                    w, h = tf.exp(tf.minimum(t[..., 0], 4.0)), tf.exp(tf.minimum(t[..., 1], 4.0))
                elif WH_SQRT:
                    w, h = tf.square(t[..., 0]) * GRID_W, tf.square(t[..., 1]) * GRID_H
                else:
                    w, h = tf.nn.relu(t[..., 0]), tf.nn.relu(t[..., 1])
                return t[..., 2] - w / 2, t[..., 3] - h / 2, t[..., 2] + w / 2, t[..., 3] + h / 2
            px1, py1, px2, py2 = to_xyxy(pred)
            gx1, gy1, gx2, gy2 = to_xyxy(tgt)
            inter = tf.nn.relu(tf.minimum(px2, gx2) - tf.maximum(px1, gx1)) * \
                tf.nn.relu(tf.minimum(py2, gy2) - tf.maximum(py1, gy1))
            union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
            hull = (tf.maximum(px2, gx2) - tf.minimum(px1, gx1)) * (tf.maximum(py2, gy2) - tf.minimum(py1, gy1))
            giou = inter / (union + 1e-6) - (hull - union) / (hull + 1e-6)
            m = mask[..., 0]
            loss += GIOU_WEIGHT * tf.reduce_sum((1.0 - giou) * m, axis=[1, 2, 3]) / n
        return loss

    return box_loss


class DetDataset(tf.keras.utils.PyDataset):
    """Augment lại mỗi epoch. Chạy song song WORKERS luồng -> GPU không phải chờ CPU.
    Mỗi batch có bộ sinh ngẫu nhiên riêng (seed theo epoch + batch) -> an toàn khi chạy đa luồng."""

    def __init__(self, samples, num_classes, batch_size=16, seed=0, **kw):
        super().__init__(**kw)
        self.samples, self.C, self.bs, self.seed = samples, num_classes, batch_size, seed
        self.epoch = 0
        self.order = np.random.default_rng(seed).permutation(len(samples))

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.bs))

    def on_epoch_end(self):
        self.epoch += 1
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.samples))

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        X, H, B = [], [], []
        for j in self.order[i * self.bs:(i + 1) * self.bs]:
            s = self.samples[j]
            img, boxes = augment(s["img"], s["boxes"], rng)
            h, b = encode_targets(boxes, self.C)
            X.append(preprocess(img))
            H.append(h)
            B.append(b)
        return np.stack(X), (np.stack(H), np.stack(B))


# =====================================================================================
#  PHẦN 1 - DETECTOR (step5)
# =====================================================================================
MAX_PARAMS, MAX_BYTES = 200_000, 1.5 * 1024 * 1024
TFLM_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "PAD", "RESIZE_NEAREST_NEIGHBOR", "LOGISTIC",
            "QUANTIZE", "DEQUANTIZE", "CONCATENATION", "RESHAPE", "MEAN", "MUL", "SUB",
            "FULLY_CONNECTED", "MAX_POOL_2D", "AVERAGE_POOL_2D", "RELU", "RELU6", "SOFTMAX", "GATHER"}

classes = load_classes()
C = len(classes)
print(f"\n===== KIỂM TRA TRƯỚC KHI TRAIN =====\n{C} lớp: {classes}")

# (1) Tên ảnh test phải khớp sample_id của sample_submission -> sai thì mọi dòng detection = []
_sample = pd.read_csv(DATA / "sample_submission.csv", dtype=str)
_need = set(_sample.sample_id[_sample.task_type == "DETECTION"])
_have = {p.stem for p in list_images(TEST_IMG)}
print(f"Ảnh test: {len(_have)} | sample_id DETECTION cần: {len(_need)} | khớp: {len(_need & _have)}")
if _need - _have:
    raise RuntimeError(f"Thiếu ảnh test cho {len(_need - _have)} sample_id, ví dụ {sorted(_need - _have)[:3]}; "
                       f"tên ảnh test có dạng {sorted(_have)[:3]}")

# (2) Định dạng nhãn: in thô 3 file đầu + tự nhận dạng (dừng nếu không chắc chắn)
_all_train = list_images(TRAIN_IMG)
for p in _all_train[:3]:
    lp = TRAIN_LBL / (p.stem + ".txt")
    raw = lp.read_text(encoding="utf-8").splitlines()[:4] if lp.exists() else ["(KHÔNG CÓ FILE)"]
    print(f"  {lp.name}: {raw}")
if LABEL_FORMAT == "auto":
    LABEL_FORMAT = detect_label_format(_all_train, C)
print(f"=> Định dạng nhãn: {LABEL_FORMAT}")

# (3) Vẽ box lên vài ảnh -> mở tab Output xem bằng mắt box có ôm đúng vật không
_prev = OUT / "label_preview"
_prev.mkdir(exist_ok=True)
for p in _all_train[:6]:
    s = load_samples([p], TRAIN_LBL, LABEL_FORMAT, C)[0]
    im = cv2.cvtColor(s["img"], cv2.COLOR_RGB2BGR)
    Hc, Wc = im.shape[:2]
    for c, x1, y1, x2, y2 in s["boxes"]:
        cv2.rectangle(im, (int(x1 * Wc), int(y1 * Hc)), (int(x2 * Wc), int(y2 * Hc)), (0, 0, 255), 2)
        cv2.putText(im, classes[int(c)], (int(x1 * Wc) + 2, int(y1 * Hc) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(_prev / f"{p.stem}.jpg"), im)
print(f"Đã vẽ box lên 6 ảnh: {_prev}  (MỞ RA XEM trước khi chạy thật!)")

train_paths, val_paths = split_train_val(seed=SEED)
if TRAIN_ON_ALL:
    train_paths = train_paths + val_paths
if QUICK_TEST:
    train_paths, val_paths, EPOCHS, EVAL_EVERY = train_paths[:300], val_paths[:100], 2, 1
    print("*** QUICK_TEST: 300 ảnh train, 100 ảnh val, 2 epoch - CHỈ để kiểm tra pipeline ***")
print("\n===== DETECTOR =====\nĐang nạp ảnh...")
t0 = time.time()
train_s = load_samples(train_paths, TRAIN_LBL, LABEL_FORMAT, C)
val_s = load_samples(val_paths, TRAIN_LBL, LABEL_FORMAT, C)
_n_box = np.bincount(np.concatenate([s["boxes"][:, 0] for s in train_s]).astype(int), minlength=C)
print("Số box train mỗi lớp:", dict(zip(classes, _n_box.tolist())))
test_paths = list_images(TEST_IMG)
print(f"Train {len(train_s)} | Val {len(val_s)} | Test {len(test_paths)} | đầu vào {INPUT_W}x{INPUT_H} "
      f"| nạp mất {time.time() - t0:.0f}s")
X_val = np.stack([preprocess(s["img"]) for s in val_s])


def evaluate(predict_fn, samples, X, iou_thr=0.5):
    heat, box = predict_fn(X)
    preds = {s["id"]: decode(heat[i], box[i], s["orig_size"]) for i, s in enumerate(samples)}
    return map50(preds, {s["id"]: gt_pixels(s) for s in samples}, C, iou_thr=iou_thr, return_per_class=True)


class EMA(tf.keras.callbacks.Callback):
    """Giữ bản trung bình trượt của mọi trọng số (kể cả thống kê BatchNorm).
    decay tăng dần lúc đầu: min(EMA_DECAY, (1+step)/(10+step)) -> không bị kéo về trọng số khởi tạo."""

    def __init__(self, decay):
        super().__init__()
        self.decay = decay

    def on_train_begin(self, logs=None):
        self.vars = [v for v in self.model.weights if "float" in str(v.dtype)]
        self.shadow = [tf.Variable(tf.convert_to_tensor(v), trainable=False) for v in self.vars]
        self.backup = [tf.Variable(tf.convert_to_tensor(v), trainable=False) for v in self.vars]
        self.step = tf.Variable(0.0, trainable=False)

        @tf.function
        def _update():
            self.step.assign_add(1.0)
            d = tf.minimum(self.decay, (1.0 + self.step) / (10.0 + self.step))
            for s, v in zip(self.shadow, self.vars):
                s.assign(d * s + (1.0 - d) * tf.cast(v, s.dtype))
        self._update = _update

    def on_train_batch_end(self, batch, logs=None):
        self._update()

    def swap_in(self):    # trọng số đang train -> backup, EMA -> model
        for b, s, v in zip(self.backup, self.shadow, self.vars):
            b.assign(v)
            v.assign(s)

    def swap_out(self):   # trả lại trọng số đang train
        for b, v in zip(self.backup, self.vars):
            v.assign(b)


class MapCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, path, ema=None):
        super().__init__()
        self.path, self.best, self.best_epoch, self.t = path, -1.0, 0, time.time()
        self.ref, self.ref_epoch = -1.0, 0   # mốc để tính kiên nhẫn (phải vượt MIN_DELTA)
        self.ema = ema

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % EVAL_EVERY and epoch + 1 != EPOCHS:
            return
        if self.ema is not None:
            self.ema.swap_in()   # đánh giá + lưu bằng trọng số EMA
        m, _ = evaluate(lambda x: self.model.predict(x, batch_size=64, verbose=0), val_s, X_val)
        mark = ""
        if m >= self.best:
            self.best, self.best_epoch, mark = m, epoch + 1, "  <- tốt nhất, đã lưu"
            self.model.save_weights(self.path)
        if self.ema is not None:
            self.ema.swap_out()
        per_ep = (time.time() - self.t) / (epoch + 1)
        note = ""
        if PATIENCE > 0:
            ep = epoch + 1
            if m >= self.ref + MIN_DELTA:
                self.ref, self.ref_epoch, self.lr_epoch = m, ep, ep
            opt = self.model.optimizer
            lr = float(opt.learning_rate.numpy())
            if ep - max(self.ref_epoch, getattr(self, "lr_epoch", 0)) >= PLATEAU and lr > LR * MIN_LR_FRAC * 1.001:
                lr = max(lr * 0.5, LR * MIN_LR_FRAC)
                opt.learning_rate.assign(lr)
                self.lr_epoch = ep
                note += f" | LR -> {lr:.2e}"
            if ep - self.ref_epoch >= PATIENCE:
                self.model.stop_training = True
                note += f" | DỪNG SỚM: {PATIENCE} epoch không tiến bộ"
            note += f" | lr {lr:.1e}"
        print(f"  epoch {epoch + 1:>3}: loss {logs['loss']:.3f} | val mAP50 {m:.3f}{mark} "
              f"| {per_ep:.1f}s/epoch, tối đa còn ~{per_ep * (EPOCHS - epoch - 1) / 60:.0f} phút{note}", flush=True)


model, export_core = build_model(C)
n_params = model.count_params()
print(f"Số tham số: {n_params:,} (giới hạn {MAX_PARAMS:,})")
steps = int(np.ceil(len(train_s) / BATCH)) * EPOCHS
if PATIENCE > 0:
    assert not TRAIN_ON_ALL, "TRAIN_ON_ALL thì val nằm trong train -> không dùng dừng sớm; đặt PATIENCE=0, EPOCHS=epoch tốt nhất"
    _lr = LR
else:
    _lr = tf.keras.optimizers.schedules.CosineDecay(LR, steps, alpha=0.05)
model.compile(optimizer=tf.keras.optimizers.Adam(_lr),
              loss=[focal_loss, make_box_loss(C)],
              # Tắt XLA: trên GPU Keras 3 tự bật XLA, mà XLA không có đạo hàm của resize nearest
              # (ResizeNearestNeighborGrad). Không ảnh hưởng mô hình hay file TFLite.
              jit_compile=False)
ema = EMA(EMA_DECAY) if EMA_DECAY > 0 else None
ckpt = MapCheckpoint(str(OUT / "tiny_det_best.weights.h5"), ema)
print(f"Run A: SPLIT_BOX={SPLIT_BOX} WH_SQRT={WH_SQRT} CALIB_N={CALIB_N} GIOU_WEIGHT={GIOU_WEIGHT} "
      f"EMA_DECAY={EMA_DECAY} | EPOCHS={EPOCHS} PATIENCE={PATIENCE} TRAIN_ON_ALL={TRAIN_ON_ALL}")
t0 = time.time()
model.fit(DetDataset(train_s, C, BATCH, seed=SEED, workers=WORKERS, use_multiprocessing=False, max_queue_size=16),
          epochs=EPOCHS, verbose=0, callbacks=([ema] if ema else []) + [ckpt])
print(f"Train xong sau {(time.time() - t0) / 60:.1f} phút. Tốt nhất: epoch {ckpt.best_epoch}, val mAP50 {ckpt.best:.3f}")
model.load_weights(ckpt.path)
model.save(OUT / "tiny_det.keras")

predict_float = lambda x: model.predict(x, batch_size=64, verbose=0)
m_float, per_cls = evaluate(predict_float, val_s, X_val)
_, per_cls30 = evaluate(predict_float, val_s, X_val, iou_thr=0.3)
print(f"\nmAP50 val (float32): {m_float:.3f}")
print(f"  {'lớp':<24}{'AP@0.5':>8}{'AP@0.3':>8}  chẩn đoán")
for c, ap in per_cls.items():
    ap30 = per_cls30[c]
    hint = "box lệch" if ap30 - ap > 0.15 else ("khó thấy / nhầm lớp" if ap30 < 0.5 else "")
    print(f"  {classes[c]:<24}{ap:>8.3f}{ap30:>8.3f}  {hint}")


# ---- Xuất TFLite INT8 ----
_calib_idx = np.random.default_rng(SEED).permutation(len(train_s))[:CALIB_N]


def representative_data():
    for j in _calib_idx:
        yield [preprocess(train_s[j]["img"])[None]]


export_in = tf.keras.Input((INPUT_H, INPUT_W, 3), batch_size=1, name="image")
export_model = tf.keras.Model(export_in, export_core(export_in, training=False), name="tiny_det_export")
conv = tf.lite.TFLiteConverter.from_keras_model(export_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
(OUT / "tiny_det_int8.tflite").write_bytes(tfl)

interp = tf.lite.Interpreter(model_content=tfl,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outs = interp.get_output_details()
in_scale, in_zp = inp["quantization"]


def _run_tflite(x):
    q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q[None])
    interp.invoke()
    return [(interp.get_tensor(d["index"])[0].astype(np.float32) - d["quantization"][1]) * d["quantization"][0]
            for d in outs]


# Thứ tự output trong TFLite không chắc giống Keras (wh và off lại cùng shape) ->
# ghép từng output Keras với output TFLite gần nó nhất trên vài ảnh val.
_ref = export_core.predict(X_val[:8], verbose=0)
_got = [_run_tflite(x) for x in X_val[:8]]
out_order = []
for k, r in enumerate(_ref):
    errs = [np.mean(np.abs(np.stack([g[j] for g in _got]) - r)) if outs[j]["shape"][-1] == r.shape[-1]
            and j not in out_order else np.inf for j in range(len(outs))]
    out_order.append(int(np.argmin(errs)))
    print(f"  output Keras {k} ({export_core.output_names[k] if hasattr(export_core, 'output_names') else k}) "
          f"-> TFLite {out_order[-1]} | sai số TB {min(errs):.4f}")
heat_d = outs[out_order[0]]
box_ds = [outs[j] for j in out_order[1:]]   # [box] hoặc [wh, off]


def tflite_predict(X):
    heats, boxes = [], []
    for x in X:
        o = _run_tflite(x)
        heats.append(o[out_order[0]])
        if SPLIT_BOX:
            wh, off = o[out_order[1]], o[out_order[2]]
            g = wh.shape[:2]
            boxes.append(np.concatenate([wh.reshape(*g, C, 2), off.reshape(*g, C, 2)], -1).reshape(*g, 4 * C))
        else:
            boxes.append(o[out_order[1]])
    return np.stack(heats), np.stack(boxes)


m_int8, _ = evaluate(tflite_predict, val_s, X_val)

# ---- Kiểm tra ràng buộc ----
try:
    ops = sorted({o["op_name"] for o in interp._get_ops_details()})
except Exception:
    ops = []
bad_ops = [o for o in ops if o not in TFLM_OPS]
in_shape = tuple(inp["shape"][1:3])
checks = [
    (f"Đầu vào {in_shape[1]}x{in_shape[0]} (<= 320x240)", in_shape[1] <= 320 and in_shape[0] <= 240),
    (f"Số tham số {n_params:,} (<= {MAX_PARAMS:,})", n_params <= MAX_PARAMS),
    (f"Dung lượng {len(tfl) / 1024:.0f} KB (<= 1536 KB)", len(tfl) <= MAX_BYTES),
    (f"Kiểu dữ liệu INT8 (vào {inp['dtype'].__name__}, ra {heat_d['dtype'].__name__})",
     inp["dtype"] == np.int8 and heat_d["dtype"] == np.int8),
    (f"Op TFLite Micro: {', '.join(ops) or '(không đọc được)'}", not bad_ops),
]
print("\n===== KIỂM TRA RÀNG BUỘC =====")
for text, ok in checks:
    print(f"  [{'OK' if ok else 'LỖI'}] {text}")
if bad_ops:
    print(f"  Op không chắc có trong TFLite Micro: {bad_ops}")
print(f"\nmAP50 val: float32 {m_float:.3f} -> INT8 {m_int8:.3f} (mất {m_float - m_int8:+.3f} do lượng tử hóa)")
print(f"Ước lượng điểm detection: 0,70 x {m_int8 * 100:.1f} = {0.7 * m_int8 * 100:.1f}")

(OUT / "tiny_det_quant.json").write_text(json.dumps({
    "input_w": INPUT_W, "input_h": INPUT_H, "num_classes": C, "classes": classes,
    "input_scale": float(in_scale), "input_zero_point": int(in_zp),
    "heat_scale": float(heat_d["quantization"][0]), "heat_zero_point": int(heat_d["quantization"][1]),
    "box_outputs": [{"name": n, "scale": float(d["quantization"][0]), "zero_point": int(d["quantization"][1])}
                    for n, d in zip(["wh", "off"] if SPLIT_BOX else ["box"], box_ds)],
    "preprocess": "resize RGB -> x = pixel/127.5 - 1 -> q = round(x/input_scale + input_zero_point)",
    "box_layout": ("wh[y][x][2c..2c+1] = sqrt(w/W), sqrt(h/H) nếu wh_sqrt; off[y][x][2c..2c+1] = dx, dy (ô lưới)"
                   if SPLIT_BOX else "box[y][x][4c..4c+3] = w, h, dx, dy (ô lưới); w = exp(w), h = exp(h) nếu wh_log"),
    "wh_log": WH_LOG, "wh_sqrt": WH_SQRT, "split_box": SPLIT_BOX, "stride": STRIDE,
}, indent=2, ensure_ascii=False), encoding="utf-8")

# ---- Dự đoán test bằng mô hình INT8 ----
test_s = load_samples(test_paths, None)
X_test = np.stack([preprocess(s["img"]) for s in test_s])
heat, box = tflite_predict(X_test)
det = {s["id"]: decode(heat[i], box[i], s["orig_size"]) for i, s in enumerate(test_s)}
(OUT / "det_predictions.json").write_text(json.dumps(det), encoding="utf-8")
print(f"Đã lưu det_predictions.json ({sum(map(len, det.values()))} box, từ mô hình INT8)")


# =====================================================================================
#  PHẦN 2 - INTENT TextCNN (step2, chỉ train mô hình cuối, bỏ phần CV cho nhanh)
# =====================================================================================
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Vocab:
    def __init__(self, texts):
        words = {w for t in texts for w in normalize(t).split()}
        self.itos = ["<pad>", "<oov>"] + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.stoi.get(w, 1) for w in normalize(text).split()][:max_len]
        return ids + [0] * (max_len - len(ids))


# ---- Sinh câu train bằng TỪ ĐIỂN (intent_augment.py). Chỉ dùng khi TRAIN. ----
# Mọi tên đều lấy từ intent_train.csv vòng loại. Dự đoán: normalize -> TextCNN, KHÔNG dùng từ điển.
IN_SCOPE_NAMES = ["chai", "chai nuoc", "chai nhua", "binh nuoc", "coc", "coc uong nuoc", "ly", "tach",
                  "dien thoai", "dien thoai di dong", "smartphone", "sach", "cuon sach", "quyen sach",
                  "chuot", "chuot may tinh", "con chuot may tinh", "hop", "cai hop", "hop giay", "thung giay",
                  "oto do choi", "o to do choi", "xe oto do choi", "xe do choi"]
OUT_SCOPE_NAMES = ["binh hoa", "cai thia", "den ban", "tai nghe", "cai vi", "cay but", "cai remote", "cai dia",
                   "banh mi", "con meo", "cuon giay ve sinh", "qua chuoi", "cai mu", "dong xu", "laptop",
                   "cuc tay", "doi giay", "cai thuoc ke", "con cho", "cai bat", "cai keo", "kinh mat", "cai quat",
                   "chia khoa", "con gau bong", "o cam dien", "may tinh bang", "cay nen", "qua bong", "cai o",
                   "ban phim", "dong ho", "cai ghe", "em be", "chau cay", "con chim", "qua tao"]
_IN_SORTED = sorted(set(IN_SCOPE_NAMES), key=len, reverse=True)
_IN_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in _IN_SORTED) + r")\b")


def augment_with_dictionary(texts, labels, n_in=1, n_out=1, seed=0):
    """Câu COUNT/EXIST/LOCATION có tên vật 7 lớp -> +1 câu đổi tên khác trong 7 lớp (giữ nhãn)
    + 1 câu đổi sang vật ngoài 7 lớp (OUT_OF_SCOPE). Chỉ gọi trên phần TRAIN."""
    rng = random.Random(seed)
    out_t, out_l = [], []
    for t, l in zip(texts, labels):
        t = normalize(t)
        out_t.append(t)
        out_l.append(l)
        if l not in ("COUNT", "EXIST", "LOCATION") or not _IN_RE.search(t):
            continue
        for _ in range(n_in):
            new = _IN_RE.sub(lambda m: rng.choice(_IN_SORTED), t, count=1)
            if new != t:
                out_t.append(new)
                out_l.append(l)
        for _ in range(n_out):
            out_t.append(_IN_RE.sub(lambda m: rng.choice(OUT_SCOPE_NAMES), t, count=1))
            out_l.append("OUT_OF_SCOPE")
    return out_t, out_l


def build_textcnn(vocab_size):
    tin = L.Input((MAX_LEN,), dtype="int32", name="tokens")
    e = L.Embedding(vocab_size, 32)(tin)
    br = [L.GlobalMaxPooling1D()(L.Conv1D(32, k, activation="relu", padding="same")(e)) for k in (2, 3, 4)]
    h = L.Dropout(0.3)(L.Concatenate()(br))
    m = tf.keras.Model(tin, L.Dense(len(INTENTS), activation="softmax")(h))
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train_intent(texts, labels):
    """(sinh câu) -> vocab -> mã hóa -> train. Trả về (mô hình, vocab, X train, y train)."""
    t, l = augment_with_dictionary(texts, labels, seed=SEED) if DICT_AUGMENT else (list(texts), list(labels))
    vocab = Vocab(t)
    X = np.array([vocab.encode(q, MAX_LEN) for q in t], np.int32)
    y = np.array([label2id[v] for v in l], np.int32)
    present = np.unique(y)
    cw = dict(zip(present.tolist(), (len(y) / (len(present) * np.bincount(y)[present])).tolist()))
    m = build_textcnn(len(vocab))
    m.fit(X, y, epochs=INTENT_EPOCHS, batch_size=INTENT_BATCH, class_weight=cw, verbose=0, shuffle=True)
    return m, vocab, X, y


intent = {}
if RUN_INTENT and (DATA / "intent_train.csv").exists():
    print("\n===== INTENT =====")
    tf.keras.utils.set_random_seed(SEED)
    itrain = pd.read_csv(DATA / "intent_train.csv")
    itest = pd.read_csv(DATA / "intent_test.csv")
    label2id = {l: i for i, l in enumerate(INTENTS)}

    # Kiểm tra nhanh: giữ lại 20% câu GỐC làm val; sinh câu chỉ trên 80% còn lại (không rò rỉ)
    _rng = np.random.default_rng(SEED)
    _idx = _rng.permutation(len(itrain))
    _va, _tr = _idx[:len(itrain) // 5], _idx[len(itrain) // 5:]
    m_chk, v_chk, _, _ = train_intent(itrain.question.iloc[_tr], itrain.intent.iloc[_tr])
    Xv = np.array([v_chk.encode(q, MAX_LEN) for q in itrain.question.iloc[_va]], np.int32)
    yv = itrain.intent.iloc[_va].map(label2id).values
    print(f"Holdout 20% (câu gốc): accuracy {(m_chk.predict(Xv, verbose=0).argmax(1) == yv).mean():.4f}")

    # Mô hình cuối: toàn bộ 6.000 câu
    imodel, vocab, Xi, yi = train_intent(itrain.question, itrain.intent)
    print(f"{len(itrain)} câu gốc -> {len(Xi)} câu train (DICT_AUGMENT = {DICT_AUGMENT}) | "
          f"phân bố {dict(zip(INTENTS, np.bincount(yi, minlength=5).tolist()))}")
    print(f"Tham số {imodel.count_params():,} | vocab {len(vocab)}")
    imodel.save(OUT / "intent_textcnn.keras")
    (OUT / "intent_vocab.txt").write_text("\n".join(vocab.itos), encoding="utf-8")

    Xt = np.array([vocab.encode(q, MAX_LEN) for q in itest.question], np.int32)
    probs = imodel.predict(Xt, verbose=0)
    itest["intent"] = [INTENTS[i] for i in probs.argmax(1)]
    itest[["question_id", "intent"]].to_csv(OUT / "intent_predictions.csv", index=False)
    intent = dict(zip(itest.question_id.astype(str), itest.intent))
    print("Phân bố dự đoán test:", itest.intent.value_counts().to_dict(), "(test cân bằng: mỗi nhãn 800)")

    try:
        def rep_text():
            for i in range(0, len(Xi), 4):
                yield [Xi[i:i + 1]]
        ic = tf.lite.TFLiteConverter.from_keras_model(imodel)
        ic.optimizations = [tf.lite.Optimize.DEFAULT]
        ic.representative_dataset = rep_text
        ic.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        ic.inference_output_type = tf.int8
        itfl = ic.convert()
        (OUT / "intent_int8.tflite").write_bytes(itfl)
        print(f"TFLite INT8 intent: {len(itfl) / 1024:.1f} KB")
    except Exception as ex:
        print(f"[Bỏ qua xuất TFLite intent] {type(ex).__name__}: {ex}")


# =====================================================================================
#  PHẦN 3 - GHÉP submission.csv (step4)
# =====================================================================================
print("\n===== SUBMISSION =====")
sample = pd.read_csv(DATA / "sample_submission.csv", dtype=str)


def clean_boxes(boxes):
    good = []
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b["bbox"])
        s, c = float(b["score"]), int(b["class_id"])
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, s)) or x2 <= x1 or y2 <= y1 or not 0 <= c < C:
            continue
        good.append({"class_id": c, "score": round(min(max(s, 0.0), 1.0), 4),
                     "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    good.sort(key=lambda b: -b["score"])
    return good[:100]


rows, errors, missing_det, missing_int = [], [], 0, 0
for task, sid in zip(sample.task_type, sample.sample_id):
    if task == "DETECTION":
        missing_det += sid not in det
        pred = json.dumps(clean_boxes(det.get(sid, [])), separators=(",", ":"))
    elif task == "INTENT":
        missing_int += sid not in intent
        pred = intent.get(sid, "COUNT")
    else:
        errors.append(f"task_type lạ: {task}")
        continue
    rows.append((task, sid, pred))

SUB_FILE = OUT / "submission.csv"
with open(SUB_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task_type", "sample_id", "prediction"])
    w.writerows(rows)

check = pd.read_csv(SUB_FILE, dtype=str, keep_default_na=False)
if len(check) != len(sample):
    errors.append(f"Số dòng {len(check)} != {len(sample)}")
if check.sample_id.duplicated().any():
    errors.append("Có sample_id trùng")
if set(check.sample_id) != set(sample.sample_id):
    errors.append("Danh sách sample_id khác file mẫu")
if missing_det:
    errors.append(f"{missing_det} ảnh test không có dự đoán (tên ảnh không khớp sample_id?)")
if missing_int:
    errors.append(f"{missing_int} câu intent không có dự đoán")
if QUICK_TEST:
    errors.append("QUICK_TEST = True: đây là bản CHẠY THỬ (2 epoch) - đặt QUICK_TEST = False rồi chạy lại để nộp")
print(f"Detection: {(check.task_type == 'DETECTION').sum()} ảnh (thiếu {missing_det}) | "
      f"Intent: {(check.task_type == 'INTENT').sum()} câu (thiếu {missing_int})")
if errors:
    print("LỖI - CHƯA NỘP ĐƯỢC:", *errors[:20], sep="\n  - ")
else:
    print(f"OK - file hợp lệ, sẵn sàng nộp: {SUB_FILE}")
