"""BƯỚC 3 - Detection baseline bằng YOLO pretrained (chạy GPU RTX 5060).

Chạy:  python step3_yolo_baseline.py
Kết quả: outputs/det_predictions_yolo.json  -> CHỈ để tham khảo, KHÔNG dùng để nộp

MỤC ĐÍCH: lấy "điểm trần" tham chiếu. Mô hình này (~2,6 triệu tham số, ảnh 960px) KHÔNG đạt
ràng buộc ESP32 -> bài nộp dùng step5_tiny_detector.py. Dùng cùng tập val với step5
(seed 42, 20%) nên mAP hai bên so sánh trực tiếp được.

Trên Windows bắt buộc có `if __name__ == "__main__":` vì DataLoader dùng nhiều process.
"""
import json
import random
import shutil

from PIL import Image

from common import OUT, TEST_IMG, TRAIN_IMG, TRAIN_LBL, list_images, load_classes

SEED = 42
VAL_RATIO = 0.2
MODEL = "yolo11n.pt"   # pretrained COCO, tự tải lần đầu
IMGSZ = 960            # ảnh gốc 1672x941 có vật rất nhỏ -> dùng ảnh lớn hơn 640
EPOCHS = 150
BATCH = 8              # đủ cho 8 GB VRAM ở 960px; báo lỗi out of memory thì giảm về 4
CONF = 0.001           # ngưỡng thấp: mAP lợi khi giữ nhiều box (tối đa 100/ảnh)
FULL_DATA = False      # True = train lại trên 100% ảnh (dùng khi đã chọn xong cấu hình)

YOLO_DS = OUT / "yolo_ds"


def build_dataset(classes):
    """Chia train/val (giống hệt common.split_train_val) rồi copy sang cấu trúc Ultralytics cần."""
    imgs = list_images(TRAIN_IMG)
    random.Random(SEED).shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_RATIO))
    splits = {"val": imgs[:n_val], "train": imgs if FULL_DATA else imgs[n_val:]}

    if YOLO_DS.exists():
        shutil.rmtree(YOLO_DS)
    for split, files in splits.items():
        (YOLO_DS / split / "images").mkdir(parents=True)
        (YOLO_DS / split / "labels").mkdir(parents=True)
        for img in files:
            shutil.copy(img, YOLO_DS / split / "images" / img.name)
            lbl = TRAIN_LBL / (img.stem + ".txt")
            dst = YOLO_DS / split / "labels" / (img.stem + ".txt")
            shutil.copy(lbl, dst) if lbl.exists() else dst.write_text("")

    yaml = YOLO_DS / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    yaml.write_text(f"path: {YOLO_DS.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n{names}\n", encoding="utf-8")
    print(f"Train {len(splits['train'])} ảnh | Val {len(splits['val'])} ảnh")
    return yaml


def main():
    from ultralytics import YOLO

    classes = load_classes()
    yaml = build_dataset(classes)

    model = YOLO(MODEL)
    model.train(
        data=str(yaml), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, patience=40,
        seed=SEED, project=str(OUT / "runs"), name="yolo_baseline", exist_ok=True,
        workers=4, device=0,
        # Augmentation - dữ liệu chỉ 140 ảnh nên rất cần
        fliplr=0.5, mosaic=1.0, scale=0.5, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
        close_mosaic=15,
    )
    best = YOLO(OUT / "runs" / "yolo_baseline" / "weights" / "best.pt")
    metrics = best.val(data=str(yaml), imgsz=IMGSZ, split="val", verbose=True)
    print(f"\nVal mAP50 = {metrics.box.map50:.3f}  (ước lượng điểm detection; val chỉ ~28 ảnh nên dao động)")

    # Dự đoán tập test. r.boxes.xyxy đã là pixel trên ẢNH GỐC -> đúng định dạng nộp bài.
    preds = {}
    for img in list_images(TEST_IMG):
        W, H = Image.open(img).size
        r = best.predict(str(img), imgsz=IMGSZ, conf=CONF, iou=0.6, max_det=100, verbose=False)[0]
        boxes = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            if x2 > x1 and y2 > y1:
                boxes.append({"class_id": int(c), "score": round(float(s), 4),
                              "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
        preds[img.stem] = boxes
    out = OUT / "det_predictions_yolo.json"
    out.write_text(json.dumps(preds), encoding="utf-8")
    n = sum(len(b) for b in preds.values())
    print(f"Đã lưu {out}: {len(preds)} ảnh, {n} box")


if __name__ == "__main__":
    main()
