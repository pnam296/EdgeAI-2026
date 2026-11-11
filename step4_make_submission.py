"""BƯỚC 4 - Ghép kết quả thành submission.csv và KIỂM TRA trước khi nộp.

Chạy:  python step4_make_submission.py
Đọc:  outputs/det_predictions.json, outputs/intent_predictions.csv  (thiếu file nào thì dùng giá trị rỗng)
Ghi:  outputs/submission.csv

Thứ tự và danh sách sample_id lấy đúng theo sample_submission.csv.
"""
import csv
import json
import math
import sys

import pandas as pd

from common import DATA, INTENTS, OUT, load_classes

DET_FILE = OUT / "det_predictions.json"
INTENT_FILE = OUT / "intent_predictions.csv"
SUB_FILE = OUT / "submission.csv"
MAX_BOXES = 100

n_classes = len(load_classes())
sample = pd.read_csv(DATA / "sample_submission.csv", dtype=str)

det = json.loads(DET_FILE.read_text(encoding="utf-8")) if DET_FILE.exists() else {}
intent = {}
if INTENT_FILE.exists():
    df = pd.read_csv(INTENT_FILE, dtype=str)
    intent = dict(zip(df.question_id, df.intent))
if not det:
    print("CẢNH BÁO: chưa có det_predictions.json -> detection để [] (điểm detection = 0)")
if not intent:
    print("CẢNH BÁO: chưa có intent_predictions.csv -> intent để COUNT")


def clean_boxes(boxes):
    """Lọc box hỏng, ép kiểu, giữ tối đa 100 box điểm cao nhất."""
    good = []
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b["bbox"])
        s, c = float(b["score"]), int(b["class_id"])
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, s)):
            continue
        if x2 <= x1 or y2 <= y1 or not 0 <= c < n_classes:
            continue
        good.append({"class_id": c, "score": round(min(max(s, 0.0), 1.0), 4),
                     "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    good.sort(key=lambda b: -b["score"])
    return good[:MAX_BOXES]


rows, errors, missing_det, missing_int = [], [], 0, 0
for task, sid in zip(sample.task_type, sample.sample_id):
    if task == "DETECTION":
        if sid not in det:
            missing_det += 1
        pred = json.dumps(clean_boxes(det.get(sid, [])), separators=(",", ":"))
    elif task == "INTENT":
        if sid not in intent:
            missing_int += 1
        pred = intent.get(sid, "COUNT")
        if pred not in INTENTS:
            errors.append(f"{sid}: nhãn intent lạ '{pred}'")
    else:
        errors.append(f"task_type lạ: {task}")
        continue
    rows.append((task, sid, pred))

with open(SUB_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)  # csv tự thêm ngoặc kép "" quanh JSON giống file mẫu
    w.writerow(["task_type", "sample_id", "prediction"])
    w.writerows(rows)

# ---- Đọc lại file vừa ghi và kiểm tra như hệ thống chấm ----
check = pd.read_csv(SUB_FILE, dtype=str, keep_default_na=False)
if list(check.columns) != ["task_type", "sample_id", "prediction"]:
    errors.append("Sai tên cột")
if len(check) != len(sample):
    errors.append(f"Số dòng {len(check)} != {len(sample)}")
if check.sample_id.duplicated().any():
    errors.append("Có sample_id trùng")
if set(check.sample_id) != set(sample.sample_id):
    errors.append("Danh sách sample_id khác file mẫu")
for i, r in check[check.task_type == "DETECTION"].iterrows():
    try:
        assert isinstance(json.loads(r.prediction), list)
    except Exception:
        errors.append(f"Dòng {i + 2}: prediction không phải mảng JSON")

n_box = sum(len(json.loads(p)) for p in check[check.task_type == "DETECTION"].prediction)
print(f"Detection: {(check.task_type == 'DETECTION').sum()} ảnh, {n_box} box "
      f"(thiếu dự đoán: {missing_det} ảnh)")
print(f"Intent: {(check.task_type == 'INTENT').sum()} câu (thiếu: {missing_int}) | "
      f"phân bố {check[check.task_type == 'INTENT'].prediction.value_counts().to_dict()}")
if errors:
    print("\nLỖI - CHƯA NỘP ĐƯỢC:")
    for e in errors[:20]:
        print("  -", e)
    sys.exit(1)
print(f"\nOK - file hợp lệ, sẵn sàng nộp: {SUB_FILE}")
