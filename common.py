"""Cấu hình và hàm dùng chung cho mọi script.

Vào vòng loại chỉ cần thay thư mục data/ (classes.txt mới) - code giữ nguyên.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_IMG = DATA / "train" / "images"
TRAIN_LBL = DATA / "train" / "labels"
TEST_IMG = DATA / "test" / "images"

INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_classes():
    """Dòng thứ i (đếm từ 0) trong classes.txt = class_id i."""
    lines = (DATA / "classes.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def read_yolo_label(path):
    """Trả về list (class_id, cx, cy, w, h). File rỗng/không có -> []."""
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            c, cx, cy, w, h = parts
            boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def split_train_val(val_ratio=0.2, seed=42):
    """Chia ảnh train thành (train, val) CỐ ĐỊNH - mọi mô hình dùng chung một tập val
    để so sánh công bằng (YOLO baseline vs mô hình tí hon)."""
    import random

    imgs = list_images(TRAIN_IMG)
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    return imgs[n_val:], imgs[:n_val]


def yolo_to_xyxy(cx, cy, w, h, W, H):
    """YOLO chuẩn hóa -> pixel [x_min, y_min, x_max, y_max] trên ảnh gốc W x H."""
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
