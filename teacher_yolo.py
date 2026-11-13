"""MODEL THẦY YOLO (chạy trên máy có GPU NVIDIA, vd RTX 5060) - CHỈ dùng lúc train, KHÔNG nộp.

Mục đích:
  1. Biết "mAP trần" của dữ liệu: model lớn, không giới hạn tham số, chấm trên ĐÚNG tập val
     của model tí hon (seed 42, 20%) bằng ĐÚNG hàm mAP (metrics.py) -> so sánh trực tiếp.
  2. Sinh nhãn mềm cho distillation: dự đoán của thầy trên ảnh train -> file JSON nhỏ,
     đưa lên Kaggle/Colab cho model tí hon học.

Chạy (trong venv của repo, có data/):
  python teacher_yolo.py                 # fold "all": train trên 4.800 ảnh train, chấm 1.200 ảnh val
  python teacher_yolo.py --fold 0        # distillation 2 fold: train trên nửa còn lại của tập train,
  python teacher_yolo.py --fold 1        #   dự đoán nửa CHƯA THẤY -> nhãn mềm "trung thực"
  python teacher_yolo.py --predict-only  # bỏ qua train, dùng best.pt đã có

Kết quả: outputs/teacher/<tên run>/  (best.pt, teacher_val.json, teacher_train.json, log)
Định dạng JSON: {image_id: [[class_id, score, x1, y1, x2, y2], ...]}, tọa độ CHUẨN HÓA 0..1.
Trên Windows bắt buộc có `if __name__ == "__main__":` vì DataLoader dùng nhiều process.
"""
import argparse
import json
import random
import shutil
import time

from PIL import Image

from common import OUT, TRAIN_IMG, TRAIN_LBL, load_classes, read_yolo_label, split_train_val, yolo_to_xyxy
from metrics import map50

SEED = 42
MODEL = "yolo11s.pt"   # pretrained COCO (~9,4 triệu tham số), tự tải lần đầu
IMGSZ = 640            # ảnh gốc 320x240 -> phóng x2 giúp vật nhỏ (toy_car ~12 px)
EPOCHS = 120
PATIENCE = 30
BATCH = 16             # báo lỗi out of memory thì giảm về 8
CONF = 0.001           # ngưỡng thấp: mAP lợi khi giữ nhiều box (tối đa 100/ảnh)
SAVE_CONF = 0.01       # chỉ lưu box từ ngưỡng này vào JSON nhãn mềm (file nhỏ hơn)


def fold_split(fold):
    """Trả về (ảnh train của thầy, ảnh val để chấm, ảnh cần dự đoán làm nhãn mềm)."""
    train, val = split_train_val(seed=SEED)            # giống hệt model tí hon
    if fold == "all":
        return train, val, train
    shuf = train[:]
    random.Random(SEED + 1).shuffle(shuf)
    halves = [sorted(shuf[: len(shuf) // 2]), sorted(shuf[len(shuf) // 2:])]
    k = int(fold)
    return halves[1 - k], val, halves[k]               # train nửa kia, dự đoán nửa chưa thấy


def build_dataset(ds, train, val, classes):
    if ds.exists():
        shutil.rmtree(ds)
    for split, files in (("train", train), ("val", val)):
        (ds / split / "images").mkdir(parents=True)
        (ds / split / "labels").mkdir(parents=True)
        for img in files:
            shutil.copy(img, ds / split / "images" / img.name)
            lbl = TRAIN_LBL / (img.stem + ".txt")
            dst = ds / split / "labels" / (img.stem + ".txt")
            shutil.copy(lbl, dst) if lbl.exists() else dst.write_text("")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    yaml = ds / "data.yaml"
    yaml.write_text(f"path: {ds.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n{names}\n",
                    encoding="utf-8")
    return yaml


def predict(model, imgs):
    """{id: [[cls, score, x1, y1, x2, y2] chuẩn hóa]} cho mọi box conf >= CONF."""
    out = {}
    for k in range(0, len(imgs), 64):
        chunk = imgs[k:k + 64]
        for img, r in zip(chunk, model.predict([str(p) for p in chunk], imgsz=IMGSZ, conf=CONF, iou=0.6,
                                               max_det=100, verbose=False, device=0)):
            H, W = r.orig_shape
            boxes = []
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                x1, x2 = max(0.0, x1) / W, min(float(W), x2) / W
                y1, y2 = max(0.0, y1) / H, min(float(H), y2) / H
                if x2 > x1 and y2 > y1:
                    boxes.append([int(c), round(float(s), 4), round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)])
            out[img.stem] = boxes
    return out


def evaluate(preds, imgs, C):
    """mAP50 bằng metrics.py (giống model tí hon), tính trên tọa độ pixel ảnh gốc."""
    gts, pp = {}, {}
    for img in imgs:
        W, H = Image.open(img).size
        gts[img.stem] = [(c, list(yolo_to_xyxy(cx, cy, w, h, W, H)))
                         for c, cx, cy, w, h in read_yolo_label(TRAIN_LBL / (img.stem + ".txt"))]
        pp[img.stem] = [{"class_id": c, "score": s, "bbox": [x1 * W, y1 * H, x2 * W, y2 * H]}
                        for c, s, x1, y1, x2, y2 in preds[img.stem]]
    return map50(pp, gts, C, iou_thr=0.5, return_per_class=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="all", choices=["all", "0", "1"])
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--merge", action="store_true", help="gộp nhãn mềm fold 0 + fold 1 -> teacher_soft.json")
    args = ap.parse_args()

    if args.merge:
        merged = {}
        for k in ("0", "1"):
            f = OUT / "teacher" / f"{MODEL.split('.')[0]}_{IMGSZ}_fold-{k}" / "teacher_train.json"
            part = json.loads(f.read_text(encoding="utf-8"))
            assert not merged.keys() & part.keys(), "2 fold bị trùng ảnh"
            merged.update(part)
        train, _ = split_train_val(seed=SEED)
        assert set(merged) == {p.stem for p in train}, "nhãn mềm không phủ đúng tập train"
        f = OUT / "teacher" / "teacher_soft.json"
        f.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")
        print(f"Đã gộp: {f} ({len(merged)} ảnh, {f.stat().st_size / 1e6:.1f} MB)")
        return

    from ultralytics import YOLO

    classes = load_classes()
    C = len(classes)
    name = f"{MODEL.split('.')[0]}_{IMGSZ}_fold-{args.fold}"
    run_dir = OUT / "teacher" / name
    train, val, to_predict = fold_split(args.fold)
    print(f"Run {name} | thầy train {len(train)} ảnh | val {len(val)} ảnh | nhãn mềm cho {len(to_predict)} ảnh")

    if not args.predict_only:
        yaml = build_dataset(OUT / "teacher" / f"ds_fold-{args.fold}", train, val, classes)
        t0 = time.time()
        YOLO(MODEL).train(
            data=str(yaml), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, patience=PATIENCE, seed=SEED,
            project=str(OUT / "teacher"), name=name, exist_ok=True, workers=4, device=0, cache=True,
            fliplr=0.5, mosaic=1.0, scale=0.5, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, close_mosaic=10,
            plots=False,
        )
        print(f"Train xong sau {(time.time() - t0) / 60:.1f} phút")

    best = YOLO(run_dir / "weights" / "best.pt")
    pv = predict(best, val)
    m, per = evaluate(pv, val, C)
    print(f"\n===== THẦY {name} =====")
    print(f"mAP50 val (metrics.py, cùng tập val với model tí hon): {m:.4f}")
    for c, ap_ in per.items():
        print(f"  {classes[c]:<10}{ap_:.3f}")
    (run_dir / "teacher_val.json").write_text(json.dumps(pv), encoding="utf-8")

    pt = predict(best, to_predict)
    pt = {k: [b for b in v if b[1] >= SAVE_CONF] for k, v in pt.items()}
    f = run_dir / "teacher_train.json"
    f.write_text(json.dumps(pt, separators=(",", ":")), encoding="utf-8")
    print(f"Nhãn mềm: {f} ({len(pt)} ảnh, {sum(map(len, pt.values()))} box, {f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
