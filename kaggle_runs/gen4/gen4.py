# gen4: TAO DU LIEU (GPU chi cho buoc dien nhan, con lai 4 nhan CPU) -> notebook train chi giai nen.
# So voi gen3:
#   + vat cat tu Open Images (oicrops: sach le / hop / chuot / coc / chai / dien thoai) va xe Hot Wheels (cat tung xe)
#   + GrabCut lay duong vien cho vat cat hinh chu nhat cua chuot / coc / chai / xe (sach, hop, dt von la hinh chu nhat)
#   + vat lan ra ngoai mep khung anh (~15%), vat cung lop dat sat nhau (~25%), de nhau ~10% (~20% so vat)
#   + trong so lop khi dan: book 1.5, mouse 1.5, bottle 1.2, box 1.2, cup 1, phone 0.7, toy_car 0.7; bo LVIS 'packet'
#   + Hot Wheels chi lay ~50% anh cho phan Roboflow (xe do choi da du); luu nhan da dien (gen4_rf_filled.tar) de dung lai
import os, sys, subprocess, random, shutil, time, json, tarfile, csv
from pathlib import Path
from multiprocessing import Pool

QUICK = os.environ.get("QUICK", "0") == "1"
N_PASTE = 160 if QUICK else 6000
RF_RATIO = 0.03 if QUICK else 0.6
HW_KEEP = 0.5
NPROC = 4
SPLIT_SEED = 42
CLASS_W = "book=1.5,mouse=1.5,bottle=1.2,box=1.2,cup=1,phone=0.7,toy_car=0.7"
GRABCUT = {"mouse", "cup", "bottle", "toy_car"}
t_all = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199", "pycocotools"])

INP = Path("/kaggle/input")
ALL_FILES, ALL_DIRS, COCO_DIRS = [], [], []
for root, dirs, files in os.walk(INP, followlinks=True):
    keep = []
    for d in dirs:
        (COCO_DIRS.append(Path(root) / d) if "coco-2017" in d.lower() else keep.append(d))
    dirs[:] = keep
    ALL_DIRS += [Path(root) / d for d in keep]
    ALL_FILES += [Path(root) / f for f in files]


def find(name):
    hits = sorted(p for p in ALL_FILES if p.name == name)
    assert hits, f"khong thay {name}"
    return hits[0]


def find_dir(name):
    hits = sorted(p for p in ALL_DIRS if p.name == name and (p / "images").is_dir())
    return hits[0] if hits else None


def find_coco():
    for c in COCO_DIRS:
        for ann in list(c.glob("annotations/instances_train2017.json")) + list(c.glob("*/annotations/instances_train2017.json")):
            if (ann.parent.parent / "train2017").is_dir():
                return ann, ann.parent.parent
    raise AssertionError(f"khong thay COCO trong {COCO_DIRS}")


TOOLS = find("tools_v3.txt").parent
COMP = find("classes.txt").parent
CLASSES = [l.strip() for l in (COMP / "classes.txt").read_text().splitlines() if l.strip()]
PROFILE = TOOLS / "ov7670_profile.json"
TMP, W = Path("/tmp/gen"), Path("/kaggle/working")
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
(TMP / "classes.txt").write_text("\n".join(CLASSES) + "\n")
MAN = {"quick": QUICK, "steps": {}, "class_weights": CLASS_W}


def step(name, t0):
    MAN["steps"][name] = round((time.time() - t0) / 60, 1)
    print(f"== {name}: {MAN['steps'][name]} phut", flush=True)


def mkset(d, items):
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    for ip, lp in items:
        os.symlink(ip, d / "images" / ip.name)
        dst = d / "labels" / (ip.stem + ".txt")
        shutil.copy(lp, dst) if lp.exists() else dst.write_text("")
    return d


def parallel(cmds):
    ps = [subprocess.Popen([sys.executable, *map(str, c)]) for c in cmds]
    codes = [p.wait() for p in ps]
    assert all(c == 0 for c in codes), f"loi: {codes}"


def grabcut_one(path):
    """GrabCut khoi tao bang khung (le 4%) -> mask; chi nhan khi vat chiem 20-95% khung (khong thi giu hinh chu nhat)"""
    import cv2, numpy as np
    im = cv2.imread(path)
    if im is None:
        return 0
    h, w = im.shape[:2]
    m = np.zeros((h, w), np.uint8)
    r = (max(1, int(0.04 * w)), max(1, int(0.04 * h)), w - 2 * max(1, int(0.04 * w)), h - 2 * max(1, int(0.04 * h)))
    try:
        cv2.grabCut(im, m, r, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return 0
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    frac = fg.mean() / 255
    if not 0.2 <= frac <= 0.95:
        return 0
    cv2.imwrite(path.rsplit(".", 1)[0] + "_m.png", fg)
    return 1


# ------------------------------------------------------------------ nen BTC (train 80%)
t0 = time.time()
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
btc_train = mkset(TMP / "btc_train", [(p, tr_lbl / (p.stem + ".txt")) for p in imgs[int(len(imgs) * 0.2):]])

# ------------------------------------------------------------------ Roboflow (+50% Hot Wheels): dien nhan thieu bang GPU
rf_items, hw_dir = [], None
for rname in ("rf_merged", "rf_hotwheels", "edgeai-hotwheels"):
    d = find_dir(rname)
    if d is None:
        continue
    its = [(p, d / "labels" / (p.stem + ".txt")) for p in sorted((d / "images").iterdir())]
    if "hotwheels" in rname:
        hw_dir = d
        random.Random(7).shuffle(its)
        its = its[:int(len(its) * HW_KEEP)]
    print(f"{rname}: dung {len(its)} anh", flush=True)
    MAN[f"src_{rname}"] = len(its)
    rf_items += its
random.Random(1).shuffle(rf_items)
if QUICK:
    rf_items = rf_items[:400]
n_rv = max(1, len(rf_items) // 20)
parts = {"rfv": rf_items[:n_rv], "rft": rf_items[n_rv:]}
COCO_IDS = {39: 0, 40: 1, 41: 1, 67: 2, 73: 3, 64: 4}
import torch
from ultralytics import YOLO
dev = 0 if torch.cuda.is_available() else "cpu"
filler = YOLO("yolo11x.pt")


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


n_fill = 0
for part, items in parts.items():
    d = mkset(TMP / f"{part}_filled", items)
    ims = sorted((d / "images").iterdir())
    for k in range(0, len(ims), 64):
        for r in filler.predict([str(p) for p in ims[k:k + 64]], imgsz=640, conf=0.5, classes=list(COCO_IDS),
                                verbose=False, device=dev):
            lp = d / "labels" / (Path(r.path).stem + ".txt")
            old = [list(map(float, l.split())) for l in lp.read_text().split("\n") if l.strip()]
            ob = [(o[1] - o[3] / 2, o[2] - o[4] / 2, o[1] + o[3] / 2, o[2] + o[4] / 2) for o in old]
            add = []
            for c, b in zip(r.boxes.cls.int().tolist(), r.boxes.xyxyn.tolist()):
                if all(iou(b, o) < 0.3 for o in ob):
                    add.append(f"{COCO_IDS[c]} {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {b[2] - b[0]:.6f} {b[3] - b[1]:.6f}\n")
                    ob.append(tuple(b))
            if add:
                txt = lp.read_text()
                lp.write_text(txt + ("" if not txt or txt.endswith("\n") else "\n") + "".join(add))
                n_fill += len(add)
MAN["rf_filled_boxes"] = n_fill
del filler
torch.cuda.empty_cache()
with tarfile.open(W / "gen4_rf_filled_labels.tar", "w") as tar:   # luu nhan da dien -> lan sau khoi dien lai
    for part in parts:
        tar.add(TMP / f"{part}_filled" / "labels", arcname=f"{part}_labels")
step("dien_nhan_roboflow", t0)

# ------------------------------------------------------------------ zoom + mo phong (4 phan song song)
t0 = time.time()
rft = sorted((TMP / "rft_filled" / "images").iterdir())
shards = [mkset(TMP / f"rft_in{k}", [(p.resolve(), TMP / "rft_filled" / "labels" / (p.stem + ".txt"))
                                       for p in rft[k::NPROC]]) for k in range(NPROC)]
cmds = [[TOOLS / "zoom_out.py", "--src", shards[k], "--out", TMP / f"rft_zoom{k}", "--ratio", RF_RATIO,
         "--classes", TMP / "classes.txt", "--seed", k] for k in range(NPROC)]
cmds.append([TOOLS / "zoom_out.py", "--src", TMP / "rfv_filled", "--out", TMP / "rfv_zoom", "--ratio", 1.0,
             "--classes", TMP / "classes.txt", "--seed", 99])
parallel(cmds)
cmds = [[TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / f"rft_zoom{k}", "--out",
         TMP / "rf_train" / f"p{k}", "--copies", 1, "--keep-orig", "--seed", k] for k in range(NPROC)]
cmds.append([TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / "rfv_zoom", "--out",
             TMP / "rf_val" / "p0", "--copies", 1, "--seed", 99])
parallel(cmds)
step("zoom_va_mo_phong_roboflow", t0)

# ------------------------------------------------------------------ vat cat hinh chu nhat: Open Images + Hot Wheels (+ GrabCut)
t0 = time.time()
crop_dirs = []
oi_tar = sorted(p for p in ALL_FILES if p.name == "oicrops.tar")
if oi_tar:
    with tarfile.open(oi_tar[0]) as tar:
        tar.extractall(TMP / "oi")
    shutil.copy(oi_tar[0].parent / "oicrops_index.csv", TMP / "oi" / "crops" / "index.csv")
    crop_dirs.append(TMP / "oi" / "crops")
    print("oicrops:", sum(1 for _ in csv.DictReader(open(TMP / "oi" / "crops" / "index.csv"))), "vat", flush=True)
else:
    print("(khong co oicrops)", flush=True)
if hw_dir is not None:   # cat tung xe NGUYEN VEN tu anh Hot Wheels (khong cham mep)
    import cv2
    hwc = TMP / "hw" / "toy_car"
    hwc.mkdir(parents=True)
    rows = []
    hw_imgs = sorted((hw_dir / "images").iterdir())
    random.Random(3).shuffle(hw_imgs)
    for p in hw_imgs:
        if len(rows) >= (60 if QUICK else 2500):
            break
        im = cv2.imread(str(p))
        if im is None:
            continue
        H_, W_ = im.shape[:2]
        for k, l in enumerate((hw_dir / "labels" / (p.stem + ".txt")).read_text().split("\n")):
            if not l.strip():
                continue
            c, x, y, w, h = map(float, l.split())
            x1, y1, x2, y2 = (x - w / 2) * W_, (y - h / 2) * H_, (x + w / 2) * W_, (y + h / 2) * H_
            if x1 < 2 or y1 < 2 or x2 > W_ - 2 or y2 > H_ - 2 or min(x2 - x1, y2 - y1) < 24:
                continue
            fn = hwc / f"{p.stem}_{k}.jpg"
            cv2.imwrite(str(fn), im[int(y1):int(y2), int(x1):int(x2)])
            rows.append({"file": f"toy_car/{fn.name}", "cls": "toy_car"})
    with open(TMP / "hw" / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, ["file", "cls"])
        wr.writeheader()
        wr.writerows(rows)
    crop_dirs.append(TMP / "hw")
    print("xe Hot Wheels cat rieng:", len(rows), flush=True)
gc_files = []
for cd in crop_dirs:
    for row in csv.DictReader(open(cd / "index.csv")):
        if row["cls"] in GRABCUT:
            gc_files.append(str(cd / row["file"]))
with Pool(NPROC) as pl:
    n_gc = sum(pl.map(grabcut_one, gc_files, chunksize=32))
MAN["grabcut"] = f"{n_gc}/{len(gc_files)}"
print(f"GrabCut: {n_gc}/{len(gc_files)} vat co duong vien (con lai giu hinh chu nhat)", flush=True)
step("vat_cat_oi_hw_grabcut", t0)

# ------------------------------------------------------------------ kho vat COCO + LVIS (bo packet) -> dan 4 phan -> mo phong
t0 = time.time()
ann, croot = find_coco()
lvis = []
for part in ("train", "val"):
    z = TMP / f"lvis_v1_{part}.json.zip"
    subprocess.check_call(["wget", "-q", "-O", str(z), f"https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{part}.json.zip"])
    subprocess.check_call(["unzip", "-q", "-o", str(z), "-d", str(TMP)])
    z.unlink()
    lvis.append(TMP / f"lvis_v1_{part}.json")
pool = W / "gen4_pool.pkl"
subprocess.check_call([sys.executable, str(TOOLS / "coco_paste.py"), "--ann", str(ann),
                       *[x for p in lvis for x in ("--ann", str(p))], "--images", str(croot), "--bg", str(btc_train),
                       "--out", str(TMP / "unused"), "--pool-cache", str(pool), "--only-pool"])
for p in lvis:
    p.unlink()
step("loc_kho_vat_coco_lvis", t0)
t0 = time.time()
per = N_PASTE // NPROC
crop_args = [x for cd in crop_dirs for x in ("--crops", cd)]
parallel([[TOOLS / "coco_paste.py", "--images", croot, "--bg", btc_train, "--out", TMP / f"paste{k}", "--n", per,
           "--pool-cache", pool, "--prefix", f"p{k}", "--seed", k, "--class-weights", CLASS_W, *crop_args]
          + (["--preview", W / "gen4_paste_preview.jpg"] if k == 0 else []) for k in range(NPROC)])
parallel([[TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / f"paste{k}", "--out",
           TMP / "coco" / f"p{k}", "--copies", 1, "--keep-orig", "--seed", 100 + k] for k in range(NPROC)])
step("dan_vat_va_mo_phong", t0)

# ------------------------------------------------------------------ dong goi
t0 = time.time()
for name in ("rf_train", "rf_val", "coco"):
    MAN[f"n_{name}"] = sum(1 for _ in (TMP / name).rglob("images/*"))
    with tarfile.open(W / f"gen4_{name}.tar", "w", dereference=True) as tar:
        tar.add(TMP / name, arcname=name)
step("dong_goi", t0)
MAN["tong_phut"] = round((time.time() - t_all) / 60, 1)
(W / "gen4_manifest.json").write_text(json.dumps(MAN, indent=1))
print("XONG", json.dumps(MAN), flush=True)
