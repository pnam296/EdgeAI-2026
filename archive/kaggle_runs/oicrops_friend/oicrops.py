# oicrops: cat vat NGUYEN VEN tu Open Images V7 (chi CPU) -> them sach / hop / chuot / coc / chai / dien thoai da dang.
# Open Images co co chat luong cho tung box: IsOccluded / IsTruncated / IsGroupOf / IsDepiction / IsInside
# -> chi lay vat nguyen cuon, khong bi che, khong phai nhom, khong phai hinh ve (khac sach tren ke cua COCO/LVIS).
# Output: oicrops.tar (crops/<lop>/<ImageID>_<k>.jpg, canh dai <= 256 px) + oicrops_index.csv + thong ke.
import os, sys, io, json, random, tarfile, time, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

QUICK = os.environ.get("QUICK", "0") == "1"
PER_CLASS = 60 if QUICK else 2500
MAP = {"Book": "book", "Box": "box", "Computer mouse": "mouse", "Mug": "cup", "Coffee cup": "cup",
       "Bottle": "bottle", "Mobile phone": "phone"}
BASE = "https://storage.googleapis.com/openimages"
W = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
T = Path(os.environ.get("TMP_DIR", "/tmp/oi"))
W.mkdir(parents=True, exist_ok=True)
T.mkdir(parents=True, exist_ok=True)
t0 = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
import pandas as pd, requests
from PIL import Image


def get(url, dst):
    if not dst.exists():
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for b in r.iter_content(1 << 20):
                    f.write(b)
    return dst


cd = pd.read_csv(get(f"{BASE}/v5/class-descriptions-boxable.csv", T / "classes.csv"), header=None, names=["id", "name"])
ids = dict(zip(cd.name, cd.id))
want = {ids[n]: n for n in MAP}
files = {"validation": f"{BASE}/v5/validation-annotations-bbox.csv", "test": f"{BASE}/v5/test-annotations-bbox.csv"}
if not QUICK:
    files["train"] = f"{BASE}/v6/oidv6-train-annotations-bbox.csv"
cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction",
        "IsInside"]
parts = []
for split, url in files.items():
    f = get(url, T / f"{split}.csv")
    for ch in pd.read_csv(f, usecols=cols, chunksize=2_000_000):
        ch = ch[ch.LabelName.isin(want)]
        ch = ch[(ch.IsOccluded == 0) & (ch.IsTruncated == 0) & (ch.IsGroupOf == 0) & (ch.IsDepiction == 0) & (ch.IsInside == 0)]
        a = (ch.XMax - ch.XMin) * (ch.YMax - ch.YMin)
        ch = ch[(a > 0.02) & (a < 0.7)]
        ch = ch.assign(split=split)
        parts.append(ch)
    f.unlink()
    print(f"{split}: xong ({(time.time() - t0) / 60:.1f} phut)", flush=True)
df = pd.concat(parts)
df["cls"] = df.LabelName.map(want).map(MAP)
print("box dat chuan:", df.cls.value_counts().to_dict(), flush=True)
rng = random.Random(0)
# sach: lay du x4 vi nhieu box 'Book' la CA KE SACH (loc lai bang YOLO o duoi)
sel = pd.concat([g.sample(min(len(g), PER_CLASS * (4 if c == "book" else 1)), random_state=0) for c, g in df.groupby("cls")])
print("chon:", sel.cls.value_counts().to_dict(), "| anh can tai:", sel.ImageID.nunique(), flush=True)

S3 = "https://open-images-dataset.s3.amazonaws.com"
out = T / "crops"
for c in set(MAP.values()):
    (out / c).mkdir(parents=True, exist_ok=True)
rows, fail = [], [0]
sess = requests.Session()


def work(item):
    (img_id, split), g = item
    try:
        r = sess.get(f"{S3}/{split}/{img_id}.jpg", timeout=30)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        fail[0] += 1
        return []
    w, h = im.size
    res = []
    for k, b in enumerate(g.itertuples()):
        x1, x2, y1, y2 = b.XMin * w, b.XMax * w, b.YMin * h, b.YMax * h
        p = 0.02 * max(x2 - x1, y2 - y1)
        cr = im.crop((max(0, x1 - p), max(0, y1 - p), min(w, x2 + p), min(h, y2 + p)))
        if min(cr.size) < 40:
            continue
        cr.thumbnail((256, 256))
        fn = out / b.cls / f"{img_id}_{k}.jpg"
        cr.save(fn, quality=90)
        res.append((f"{b.cls}/{fn.name}", b.cls, cr.size[0], cr.size[1]))
    return res


groups = list(sel.groupby(["ImageID", "split"]))
with ThreadPoolExecutor(32) as ex:
    for i, r in enumerate(ex.map(work, groups)):
        rows += r
        if i % 2000 == 0:
            print(f"  tai {i}/{len(groups)} anh, {len(rows)} vat, loi {fail[0]}", flush=True)
idx = pd.DataFrame(rows, columns=["file", "cls", "w", "h"])
# loc sach bang CLIP: box 'Book' cua Open Images hay la CA KE SACH / nguoi cam doc -> chi giu "1 cuon dat tren mat phang"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"])
import open_clip, torch
torch.set_num_threads(4)
clip, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
prompts = ["a photo of a single book lying on a table", "a photo of a closed book or notebook seen from above",
           "a photo of a bookshelf with many books", "a photo of a row of book spines", "a person holding and reading a book",
           "a photo of a stack of many books"]
with torch.no_grad():
    tf_ = clip.encode_text(tok(prompts))
    tf_ /= tf_.norm(dim=-1, keepdim=True)
bk = idx[idx.cls == "book"]
keep = []
for k in range(0, len(bk), 64):
    ch = bk.iloc[k:k + 64]
    ims = [Image.open(out / f).convert("RGB") for f in ch.file]
    with torch.no_grad():
        f_ = clip.encode_image(torch.stack([prep(im) for im in ims]))
        f_ /= f_.norm(dim=-1, keepdim=True)
        p_ = (100 * f_ @ tf_.T).softmax(-1)
    for f, w_, h_, pr in zip(ch.file, ch.w, ch.h, p_):
        if pr[:2].sum() > 0.6 and 0.4 <= w_ / h_ <= 2.5:   # "1 cuon" thang, khung khong qua det (ke sach dai)
            keep.append(f)
keep = keep[:PER_CLASS]
ks = set(keep)
for f in bk.file:
    if f not in ks and (out / f).exists():
        (out / f).unlink()
print(f"sach: {len(bk)} anh cat -> giu {len(keep)} (bo ke sach / nhieu cuon)", flush=True)
idx = pd.concat([idx[idx.cls != "book"], idx[idx.file.isin(keep)]])
idx.to_csv(W / "oicrops_index.csv", index=False)
with tarfile.open(W / "oicrops.tar", "w") as tar:
    tar.add(out, arcname="crops")
stat = {"vat": idx.cls.value_counts().to_dict(), "anh_loi": fail[0], "phut": round((time.time() - t0) / 60, 1),
        "quick": QUICK}
(W / "oicrops_stats.json").write_text(json.dumps(stat, indent=1))
print("XONG", stat, flush=True)
