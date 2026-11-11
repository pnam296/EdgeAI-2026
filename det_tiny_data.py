"""Phần dữ liệu của detector tí hon (chỉ numpy + OpenCV, không cần TensorFlow).

Ý TƯỞNG CenterNet ("Objects as Points"):
  Ảnh đầu vào 160x96 -> lưới đầu ra 20x12 (mỗi ô = 8x8 pixel, STRIDE = 8).
  Mỗi vật thể được đại diện bởi ĐIỂM TÂM của nó trên lưới:
    - heatmap[y, x, lớp]: "ô này có phải tâm của một vật thuộc lớp này không" (0..1),
      nhãn là một đốm Gauss quanh tâm để mô hình không bị phạt nặng khi lệch 1 ô.
    - box[y, x, lớp] = (w, h, dx, dy): kích thước vật (đơn vị: ô lưới) và độ lệch của tâm
      thật so với góc ô (0..1), chỉ học tại ô tâm (mask = 1).
      Box RIÊNG cho từng lớp: tài xế ngồi trong xe buýt có tâm trùng ô với xe buýt, nếu
      dùng chung thì 2 vật tranh nhau 1 ô và xe buýt bị dự đoán sai kích thước.
  Khi dự đoán: tìm các đỉnh trên heatmap -> mỗi đỉnh là một box. Không cần anchor, không
  cần NMS phức tạp -> rất dễ viết lại bằng C trên ESP32.
"""
import cv2
import numpy as np

from common import read_yolo_label

# ---- Cấu hình đầu vào. Cả 2 chiều phải chia hết cho 16. ----
# Warm-up: ảnh 16:9 -> 160x96.  Vòng loại: ảnh 320x240 (4:3) -> đổi thành 128x96 hoặc 160x128.
INPUT_W, INPUT_H = 160, 128
STRIDE = 8
GRID_W, GRID_H = INPUT_W // STRIDE, INPUT_H // STRIDE
CACHE_W = 640  # ảnh gốc được thu nhỏ về rộng 640 khi nạp, để crop/augment vẫn đủ nét
# Học log(w), log(h) thay vì w, h: sai 20% kích thước bị phạt NHƯ NHAU cho vật to và vật nhỏ.
# Với w, h thô, lệch 1 ô là "nhỏ" với xe tải (rộng 3 ô) nhưng "lớn" với đèn giao thông (rộng 0,6 ô).
WH_LOG = False
assert INPUT_W % 16 == 0 and INPUT_H % 16 == 0, "INPUT_W, INPUT_H phải chia hết cho 16"


def load_image(path):
    """Đọc ảnh RGB + thu nhỏ để cache. Trả về (ảnh, (W gốc, H gốc))."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # chạy được với đường dẫn tiếng Việt
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    if W0 > CACHE_W:
        img = cv2.resize(img, (CACHE_W, round(H0 * CACHE_W / W0)), interpolation=cv2.INTER_AREA)
    return img, (W0, H0)


def load_samples(img_paths, label_dir):
    """Mỗi mẫu: dict(id, img, orig_size, boxes) với boxes = mảng N x 5 [cls, x1, y1, x2, y2] chuẩn hóa 0..1."""
    samples = []
    for p in img_paths:
        img, size = load_image(p)
        rows = []
        if label_dir is not None:
            for c, cx, cy, w, h in read_yolo_label(label_dir / (p.stem + ".txt")):
                rows.append([c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        boxes = np.clip(np.array(rows, dtype=np.float32).reshape(-1, 5), [0, 0, 0, 0, 0], [999, 1, 1, 1, 1])
        samples.append({"id": p.stem, "img": img, "orig_size": size, "boxes": boxes})
    return samples


def preprocess(img_rgb):
    """Ảnh RGB uint8 bất kỳ kích thước -> float32 [-1, 1], kích thước INPUT. Dùng chung cho train/test/thiết bị."""
    x = cv2.resize(img_rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return x / 127.5 - 1.0


def augment(img, boxes, rng):
    """Tăng cường dữ liệu: crop ngẫu nhiên (zoom), lật ngang, đổi màu/sáng, đôi khi ảnh xám.
    Box đi theo ảnh; box bị cắt mất hơn 60% diện tích thì bỏ."""
    H, W = img.shape[:2]
    boxes = boxes.copy()

    if rng.random() < 0.7:  # crop ngẫu nhiên = phóng to -> vật nhỏ thành vật to hơn
        s = rng.uniform(0.55, 1.0)
        cw = s
        ch = min(1.0, s * rng.uniform(0.85, 1.15))
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

    if rng.random() < 0.5:  # lật ngang
        img = img[:, ::-1]
        if len(boxes):
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]

    img = img.astype(np.float32)
    if rng.random() < 0.8:  # độ sáng, tương phản, độ bão hòa màu
        img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.6, 1.4)
    if rng.random() < 0.1:  # camera OV7670 chất lượng thấp -> đôi khi ảnh gần như xám
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), boxes


def encode_targets(boxes, num_classes):
    """boxes chuẩn hóa [cls, x1, y1, x2, y2] -> (heatmap GH x GW x C, box GH x GW x 5C).
    box[..., 4c:4c+4] = (w, h, dx, dy) của lớp c;  box[..., 4C + c] = mask của lớp c."""
    C = num_classes
    heat = np.zeros((GRID_H, GRID_W, C), np.float32)
    box = np.zeros((GRID_H, GRID_W, 5 * C), np.float32)
    ys, xs = np.mgrid[0:GRID_H, 0:GRID_W]
    # Vật lớn ghi trước -> nếu 2 vật CÙNG LỚP trùng ô tâm, vật nhỏ (khó hơn) được giữ
    order = np.argsort(-(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])) if len(boxes) else []
    for c, x1, y1, x2, y2 in boxes[order]:
        c = int(c)
        w, h = (x2 - x1) * GRID_W, (y2 - y1) * GRID_H  # đơn vị: ô lưới
        cx, cy = (x1 + x2) / 2 * GRID_W, (y1 + y2) / 2 * GRID_H
        ix, iy = min(int(cx), GRID_W - 1), min(int(cy), GRID_H - 1)
        # Gauss elip theo hình dạng box (TTFNet) - hợp với box dẹt như vạch qua đường
        sx, sy = max(0.54 * w / 6, 0.4), max(0.54 * h / 6, 0.4)
        g = np.exp(-((xs - ix) ** 2) / (2 * sx * sx) - ((ys - iy) ** 2) / (2 * sy * sy))
        heat[:, :, c] = np.maximum(heat[:, :, c], g)
        heat[iy, ix, c] = 1.0  # đỉnh đúng bằng 1 -> được tính là "điểm dương" trong focal loss
        if WH_LOG:
            w, h = np.log(max(w, 1e-3)), np.log(max(h, 1e-3))
        box[iy, ix, 4 * c: 4 * c + 4] = [w, h, cx - ix, cy - iy]
        box[iy, ix, 4 * C + c] = 1.0
    return heat, box


def decode(heat, box, orig_size, max_det=100, min_score=0.001):
    """Đầu ra mô hình (1 ảnh) -> list box pixel trên ảnh gốc, đúng định dạng nộp bài.
    Đỉnh = ô có giá trị lớn nhất trong vùng 3x3 của chính nó (thay cho NMS)."""
    W0, H0 = orig_size
    padded = np.pad(heat, ((1, 1), (1, 1), (0, 0)), constant_values=-1)
    local_max = np.max(np.stack([padded[dy:dy + GRID_H, dx:dx + GRID_W]
                                 for dy in range(3) for dx in range(3)]), axis=0)
    peaks = (heat >= local_max) & (heat >= min_score)
    iy, ix, c = np.nonzero(peaks)
    scores = heat[iy, ix, c]
    top = np.argsort(-scores)[:max_det]
    out = []
    for i in top:
        w, h, dx, dy = box[iy[i], ix[i], 4 * c[i]: 4 * c[i] + 4]
        if WH_LOG:
            w, h = float(np.exp(min(w, 4.0))), float(np.exp(min(h, 4.0)))
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
    """Nhãn của 1 mẫu dưới dạng pixel trên ảnh gốc - dùng cho metrics.map50."""
    W0, H0 = sample["orig_size"]
    return [(int(c), [x1 * W0, y1 * H0, x2 * W0, y2 * H0]) for c, x1, y1, x2, y2 in sample["boxes"]]
