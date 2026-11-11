# Dự đoán test bằng model FLOAT (.keras) thay vì INT8 -> ghép submission.csv mới.
# Vòng loại không bắt buộc lượng tử hóa (thông báo BTC 11/11/2026).
# Dùng: python predict_float.py <thư_mục_run>   (vd kaggle_runs/A45-noaug)
#   - đọc <run>/edgeai_kaggle.py lấy đúng các hàm tiền xử lý / decode / mAP của run đó
#   - đọc <run>/output/tiny_det.keras và <run>/output/submission.csv (lấy phần intent)
#   - ghi <run>/output/submission_float.csv
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
src = (run / "edgeai_kaggle.py").read_text(encoding="utf-8").splitlines()
end = next(i for i, l in enumerate(src) if "PHẦN 1 - DETECTOR" in l) - 1   # cấu hình + hàm + lớp model (ResizeNearest), chưa train
g = {"__name__": "helpers"}
exec(compile("\n".join(src[:end]), str(run / "edgeai_kaggle.py"), "exec"), g)
np, tf, pd = g["np"], g["tf"], g["pd"]

classes = g["load_classes"]()
C = len(classes)
model = tf.keras.models.load_model(run / "output" / "tiny_det.keras", compile=False)
print(f"Model: {model.count_params():,} tham số | đầu vào {g['INPUT_W']}x{g['INPUT_H']}")
predict = lambda X: model.predict(X, batch_size=32, verbose=0)

# 1) Kiểm tra lại trên val: phải ra đúng mAP float trong log Kaggle
_, val_paths = g["split_train_val"](seed=g["SEED"])
val_s = g["load_samples"](val_paths, g["TRAIN_LBL"], "yolo", C)
X_val = np.stack([g["preprocess"](s["img"]) for s in val_s])
heat, box = predict(X_val)
preds = {s["id"]: g["decode"](heat[i], box[i], s["orig_size"]) for i, s in enumerate(val_s)}
m, per = g["map50"](preds, {s["id"]: g["gt_pixels"](s) for s in val_s}, C, iou_thr=0.5, return_per_class=True)
print(f"mAP50 val (float, chạy trên máy): {m:.4f}")
for c, ap in per.items():
    print(f"  {classes[c]:<10}{ap:.3f}")

# 2) Dự đoán test
test_s = g["load_samples"](g["list_images"](g["TEST_IMG"]), None)
det = {}
for k in range(0, len(test_s), 500):
    chunk = test_s[k:k + 500]
    heat, box = predict(np.stack([g["preprocess"](s["img"]) for s in chunk]))
    det.update({s["id"]: g["decode"](heat[i], box[i], s["orig_size"]) for i, s in enumerate(chunk)})
print(f"Test: {len(det)} ảnh, {sum(map(len, det.values()))} box")

# 3) Ghép: detection mới + intent giữ nguyên từ submission cũ
old = pd.read_csv(run / "output" / "submission.csv", dtype=str, keep_default_na=False)
sample = pd.read_csv(g["DATA"] / "sample_submission.csv", dtype=str)
assert list(old.sample_id) == list(sample.sample_id), "submission cũ lệch thứ tự file mẫu"
rows = []
for task, sid, pred in zip(old.task_type, old.sample_id, old.prediction):
    if task == "DETECTION":
        boxes = sorted(det[sid], key=lambda b: -b["score"])[:100]
        pred = json.dumps(boxes, separators=(",", ":"))
    rows.append((task, sid, pred))
out = run / "output" / "submission_float.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task_type", "sample_id", "prediction"])
    w.writerows(rows)
chk = pd.read_csv(out, dtype=str, keep_default_na=False)
assert len(chk) == len(sample) and set(chk.sample_id) == set(sample.sample_id)
assert (chk[chk.task_type == "INTENT"].prediction.values == old[old.task_type == "INTENT"].prediction.values).all()
print(f"OK -> {out}  ({(chk.task_type == 'DETECTION').sum()} detection, {(chk.task_type == 'INTENT').sum()} intent)")
