# tacocrops: cat vat THEO MASK tu TACO (anh rac thai that) -> chai nhua TRONG SUOT, coc thuy tinh / nhua / giay, hop carton.
# Chai trong suot tren nen sang la diem yeu nhat (GrabCut hay hong voi vat trong suot) -> TACO co mask san.
# Output: tacocrops.tar (crops/<lop>/<id>.jpg + <id>_m.png mask, crops/index.csv) + thong ke. Chi CPU.
import os, sys, json, tarfile, time, csv, subprocess
from pathlib import Path
from collections import defaultdict

t0 = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])
import cv2
import numpy as np
from pycocotools import mask as mu

MAP = {"Clear plastic bottle": "bottle", "Other plastic bottle": "bottle", "Glass bottle": "bottle",
       "Glass cup": "cup", "Disposable plastic cup": "cup", "Other plastic cup": "cup", "Paper cup": "cup",
       "Foam cup": "cup",
       "Corrugated carton": "box", "Other carton": "box", "Meal carton": "box", "Pizza box": "box",
       "Drink carton": "box"}
INP = Path("/kaggle/input")
ann_f = next(p for p in INP.rglob("annotations.json"))
root = ann_f.parent
W = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
out = Path(os.environ.get("TMP_DIR", "/tmp/taco")) / "crops"
d = json.load(open(ann_f))
cats = {c["id"]: c["name"] for c in d["categories"]}
print("lop TACO khop:", sorted(set(cats.values()) & set(MAP)), flush=True)
imgs = {i["id"]: i for i in d["images"]}
by = defaultdict(list)
for a in d["annotations"]:
    by[a["image_id"]].append(a)


def ioa(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if iw <= 0 or ih <= 0 else iw * ih / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


rows, stat = [], defaultdict(int)
for iid, anns in by.items():
    im_info = imgs[iid]
    sel = []
    for a in anns:
        cls = MAP.get(cats[a["category_id"]])
        if not cls or a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        iw, ih = im_info["width"], im_info["height"]
        if min(w, h) < 40 or x < 3 or y < 3 or x + w > iw - 3 or y + h > ih - 3:
            continue
        bb = (x, y, x + w, y + h)
        if any(ioa(bb, (o["bbox"][0], o["bbox"][1], o["bbox"][0] + o["bbox"][2], o["bbox"][1] + o["bbox"][3])) > 0.15
               for o in anns if o is not a):
            continue
        sel.append((a, cls, bb))
    if not sel:
        continue
    im = cv2.imread(str(root / im_info["file_name"]))
    if im is None:
        continue
    ih, iw = im.shape[:2]
    if (iw, ih) != (im_info["width"], im_info["height"]):   # anh xoay EXIF: bo qua cho an toan
        continue
    for a, cls, (x1, y1, x2, y2) in sel:
        seg = a["segmentation"]
        rles = mu.frPyObjects(seg, ih, iw) if isinstance(seg, list) else [seg]
        m = mu.decode(mu.merge(rles) if isinstance(rles, list) else rles).astype(np.uint8)
        p = int(0.03 * max(x2 - x1, y2 - y1))
        a0, b0, a1, b1 = max(0, int(x1) - p), max(0, int(y1) - p), min(iw, int(x2) + p), min(ih, int(y2) + p)
        cr, mk = im[b0:b1, a0:a1], m[b0:b1, a0:a1] * 255
        if mk.mean() / 255 < 0.15:
            continue
        s = 256 / max(cr.shape[:2])
        if s < 1:
            cr = cv2.resize(cr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            mk = cv2.resize(mk, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_NEAREST)
        (out / cls).mkdir(parents=True, exist_ok=True)
        fn = out / cls / f"taco_{a['id']}.jpg"
        cv2.imwrite(str(fn), cr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(fn).replace(".jpg", "_m.png"), mk)
        rows.append({"file": f"{cls}/{fn.name}", "cls": cls, "taco": cats[a["category_id"]]})
        stat[cats[a["category_id"]]] += 1
with open(out / "index.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, ["file", "cls", "taco"])
    wr.writeheader()
    wr.writerows(rows)
W.mkdir(parents=True, exist_ok=True)
with tarfile.open(W / "tacocrops.tar", "w") as tar:
    tar.add(out, arcname="crops")
res = {"theo_lop": {c: sum(1 for r in rows if r["cls"] == c) for c in ("bottle", "cup", "box")},
       "theo_lop_taco": dict(stat), "phut": round((time.time() - t0) / 60, 1)}
(W / "tacocrops_stats.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("XONG", res, flush=True)
