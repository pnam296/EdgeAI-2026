"""Gán nhãn NHÁP cho ảnh camera thật bằng model thầy (2.58M, chỉ dùng trên máy tính), để người sửa lại.

    python tools/prelabel.py --src data/cam_real --preview outputs/prelabel_preview.jpg

Ghi <src>/labels/*.txt (YOLO, 7 class theo data/classes.txt) và <src>/classes.txt; ảnh để nguyên chỗ cũ, được
chuyển vào <src>/images/ nếu đang nằm thẳng trong <src>. Không ghi đè nhãn đã có (người đã sửa) trừ khi --overwrite.
Ngưỡng thấp (mặc định 0.25) để thà thừa box cho người xóa còn hơn sót vật.

Sửa nhãn bằng labelImg (định dạng YOLO):  pip install labelImg  ->  labelImg data/cam_real/images data/cam_real/classes.txt data/cam_real/labels
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import yolo_reach  # noqa: E402

COLORS = [(0, 128, 255), (255, 0, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200), (128, 64, 0), (0, 0, 255)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", default=str(REPO / "kaggle_runs/cam1/out/yolo_cam_teacher.pt"))
    ap.add_argument("--dist-scale", type=float, default=1.5, help="tầm với P4 lúc train model (cam1: 1.5)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", default="", help="ảnh lưới 4x4 để xem nhanh")
    a = ap.parse_args()

    yolo_reach.apply((1.0, 1.0, a.dist_scale))
    from ultralytics import YOLO

    names = [l.strip() for l in (REPO / "data/classes.txt").read_text().splitlines() if l.strip()]
    src = Path(a.src)
    img_dir, lbl_dir = src / "images", src / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)
    for p in src.iterdir():
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            shutil.move(str(p), img_dir / p.name)
    (src / "classes.txt").write_text("\n".join(names) + "\n")

    model = YOLO(a.weights)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    tiles, n_new, counts = [], 0, np.zeros(len(names), int)
    for p in imgs:
        lp = lbl_dir / (p.stem + ".txt")
        if lp.exists() and not a.overwrite:
            continue
        r = model.predict(str(p), imgsz=320, conf=a.conf, iou=a.iou, agnostic_nms=True, verbose=False)[0]
        lines = []
        im = cv2.imread(str(p))
        for c, xywhn, xyxy, s in zip(r.boxes.cls.int().tolist(), r.boxes.xywhn.tolist(), r.boxes.xyxy.int().tolist(),
                                     r.boxes.conf.tolist()):
            lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in xywhn))
            counts[c] += 1
            cv2.rectangle(im, xyxy[:2], xyxy[2:], COLORS[c], 1)
            cv2.putText(im, f"{names[c]} {s:.2f}", (xyxy[0], max(8, xyxy[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        COLORS[c], 1)
        lp.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_new += 1
        if len(tiles) < 16:
            tiles.append(cv2.resize(im, (320, 240)))
    print(f"{len(imgs)} ảnh, gán nháp {n_new} ảnh -> {lbl_dir}")
    print("số box: " + "  ".join(f"{n} {k}" for n, k in zip(names, counts)))
    if a.preview and tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        Path(a.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.preview, grid)
        print("xem trước:", a.preview)


if __name__ == "__main__":
    main()
