"""Dán vật thật (cắt từ ảnh camera đã gán nhãn) lên ảnh camera khác -> thêm ví dụ cho lớp yếu (vd mouse).

    python tools/paste_objects.py --src outputs/cam2_data/cam_real/train --out outputs/cam2_data/paste \
        --cls mouse --contrast phone --n 400 --preview outputs/paste_preview.jpg

Chỉ dùng ảnh TRAIN làm nguồn vật và nền (không đụng tập test). Mỗi ảnh mới: 1-2 vật lớp --cls (+50% kèm 1 vật lớp
--contrast để học phân biệt), lật/xoay 90 độ, co giãn 0.85-1.2, đổi sáng +-15%, mép mờ dần để bớt vết cắt;
chỉ đặt vào chỗ không chồng lên box có sẵn.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def load(src: Path):
    items = []
    for lp in sorted((src / "labels").glob("*.txt")):
        im = cv2.imread(str(src / "images" / (lp.stem + ".png")))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = []
        for l in lp.read_text().split("\n"):
            if not l.strip():
                continue
            c, x, y, w, h = map(float, l.split())
            boxes.append((int(c), (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        items.append((lp.stem, im, boxes))
    return items


def crops(items, cls, pad=3):
    out = []
    for _, im, boxes in items:
        H, W = im.shape[:2]
        for c, x1, y1, x2, y2 in boxes:
            if c != cls or x1 < 2 or y1 < 2 or x2 > W - 2 or y2 > H - 2:   # bỏ vật bị cắt ở mép
                continue
            a, b = max(0, int(x1) - pad), max(0, int(y1) - pad)
            out.append((im[b:int(y2) + pad, a:int(x2) + pad].copy(), pad))
    return out


def augment(crop, rng):
    if rng.random() < 0.5:
        crop = cv2.flip(crop, 1)
    if rng.random() < 0.5:
        crop = cv2.flip(crop, 0)
    crop = np.rot90(crop, rng.randrange(4)).copy()
    s = rng.uniform(0.85, 1.2)
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return np.clip(crop.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)


def overlaps(b, boxes, thr=0.02):
    for _, x1, y1, x2, y2 in boxes:
        iw, ih = min(b[2], x2) - max(b[0], x1), min(b[3], y2) - max(b[1], y1)
        if iw > 0 and ih > 0 and iw * ih > thr * min((b[2] - b[0]) * (b[3] - b[1]), (x2 - x1) * (y2 - y1)):
            return True
    return False


def paste(im, boxes, crop, pad, cls, rng, tries=40):
    H, W = im.shape[:2]
    h, w = crop.shape[:2]
    if h >= H or w >= W:
        return False
    for _ in range(tries):
        x, y = rng.randrange(0, W - w), rng.randrange(0, H - h)
        b = (x + pad, y + pad, x + w - pad, y + h - pad)
        if overlaps(b, boxes):
            continue
        m = np.zeros((h, w), np.float32)
        m[pad:h - pad, pad:w - pad] = 1
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, pad / 1.5))[..., None]
        roi = im[y:y + h, x:x + w].astype(np.float32)
        im[y:y + h, x:x + w] = (m * crop + (1 - m) * roi).astype(np.uint8)
        boxes.append((cls, *b))
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", default="mouse")
    ap.add_argument("--contrast", default="phone", help="'' = không dán vật đối chứng")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = load(Path(a.src))
    cid = NAMES.index(a.cls)
    main_crops = crops(items, cid)
    con_crops = crops(items, NAMES.index(a.contrast)) if a.contrast else []
    print(f"{len(items)} ảnh nền, {len(main_crops)} vật {a.cls}, {len(con_crops)} vật {a.contrast or '-'}")
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    tiles, added = [], 0
    for i in range(a.n):
        stem, im, boxes = rng.choice(items)
        im, boxes = im.copy(), list(boxes)
        for _ in range(rng.choice((1, 1, 2))):
            added += paste(im, boxes, augment(rng.choice(main_crops)[0], rng), 3, cid, rng)
        if con_crops and rng.random() < 0.5:
            c = rng.choice(con_crops)
            paste(im, boxes, augment(c[0], rng), 3, NAMES.index(a.contrast), rng)
        H, W = im.shape[:2]
        name = f"paste_{a.cls}_{i:04d}"
        cv2.imwrite(str(out / "images" / (name + ".png")), im)
        (out / "labels" / (name + ".txt")).write_text("".join(
            f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
            for c, x1, y1, x2, y2 in boxes))
        if a.preview and len(tiles) < 8:
            t = im.copy()
            for c, x1, y1, x2, y2 in boxes:
                col = (0, 0, 255) if c == cid else (0, 255, 0) if a.contrast and c == NAMES.index(a.contrast) else (160, 160, 160)
                cv2.rectangle(t, (int(x1), int(y1)), (int(x2), int(y2)), col, 1)
            tiles.append(t)
    print(f"{a.n} ảnh, dán {added} vật {a.cls} -> {out}")
    if a.preview and tiles:
        cv2.imwrite(a.preview, np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
