"""Tạo lại DETECTION của submission bằng decode mới (DEC_NB 1, heat^2) từ trọng số đã train; giữ nguyên INTENT.
python regen.py <weights.h5> <submission_cũ.csv> <submission_mới.csv>"""
import ast, sys, csv, json, numpy as np, tensorflow as tf
from pathlib import Path
src = open("D:/edgeai/edgeai/edgeai_float.py", encoding="utf-8").read()
g = {"__name__": "x"}
exec("import os, csv, json, math, random, re, time\nimport numpy as np, cv2, tensorflow as tf\nfrom pathlib import Path", g)
g.update(INPUT_W=320, INPUT_H=240, GRID_W=80, GRID_H=60, WH_LOG=False, ALPHA=0.45, NECK_CH=64, STRIDE4=True,
         PRETRAINED=False, L=tf.keras.layers, IMG_EXTS={".png", ".jpg", ".jpeg", ".bmp"}, CACHE_W=640, C=7,
         DEC_NB=1, DEC_PW=2)
for n in ast.parse(src).body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        exec(compile(ast.Module([n], []), "f", "exec"), g)
wts, old, new = sys.argv[1:4]
m = g["build_model"](7); m.load_weights(wts)
test_s = g["load_samples"](g["list_images"](Path("D:/edgeai/edgeai/data/test/images")), None)
det_old, det_new = {}, {}
for k in range(0, len(test_s), 500):
    ch = test_s[k:k + 500]
    heat, box = m.predict(np.stack([g["preprocess"](s["img"]) for s in ch]), batch_size=32, verbose=0)
    for i, s in enumerate(ch):
        det_old[s["id"]] = g["decode"](heat[i], box[i], s["orig_size"], nb=0)
        det_new[s["id"]] = g["decode"](heat[i], box[i], s["orig_size"], nb=1)
rows = list(csv.reader(open(old, encoding="utf-8")))
# Kiểm tra: decode cũ trên máy phải khớp submission cũ (GPU) -> đúng trọng số, đúng tiền xử lý
same = tot = 0
out = [rows[0]]
for task, sid, pred in rows[1:]:
    if task == "DETECTION":
        a = json.loads(pred); b = g["clean_boxes"](det_old[sid])
        tot += 1
        same += len(a) == len(b) and all(x["class_id"] == y["class_id"] and abs(x["score"] - y["score"]) < 2e-3
                                          and max(abs(p - q) for p, q in zip(x["bbox"], y["bbox"])) < 1.0 for x, y in zip(a[:20], b[:20]))
        pred = json.dumps(g["clean_boxes"](det_new[sid]), separators=(",", ":"))
    out.append([task, sid, pred])
print(f"Khớp submission cũ (decode cũ, 20 box đầu mỗi ảnh): {same}/{tot}")
with open(new, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(out)
print("Đã ghi", new, "|", sum(r[0] == "DETECTION" for r in out[1:]), "DETECTION,", sum(r[0] == "INTENT" for r in out[1:]), "INTENT")
