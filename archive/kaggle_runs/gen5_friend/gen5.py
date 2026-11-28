# gen5: chi CPU (khong ton quota GPU). Dung lai phan Roboflow + kho vat COCO/LVIS cua gen4, chi TAO LAI phan dan vat:
#   + nen GIAY A0 TRANG tu sinh (canh chung ket) cho ~60% anh, 5-9 vat moi anh (chung ket: du do tren giay)
#   + vat cat TACO theo mask: chai nhua TRONG SUOT, chai thuy tinh, coc nhua / giay, hop carton
#   + day chuot (~60% chuot) + day gay nhieu (~15% anh) -> bot nham chuot <-> dien thoai
#   + SACH TU TAO bia nhieu mau (~40% sach dan, 60% trong do bia XANH BIEN - sach thi mau xanh): bia sac so chi ~0.5
#   + trong so: mouse 1.8, book 1.5, bottle 1.5, box 1.2, cup 1, phone 1 (gen4 de 0.7 lam tut dien thoai), toy_car 0.8
import os, sys, subprocess, random, shutil, time, json, tarfile, csv
from pathlib import Path
from multiprocessing import Pool

QUICK = os.environ.get("QUICK", "0") == "1"
N_PASTE = 160 if QUICK else 6000
NPROC = 4
SPLIT_SEED = 42
CLASS_W = "mouse=1.8,book=1.5,bottle=1.5,box=1.2,cup=1,phone=1,toy_car=0.8"
PASTE_ARGS = ["--paper-bg", 0.6, "--mouse-cable", 0.6, "--cable-distractor", 0.15, "--synth-book", 0.4, "--book-blue", 0.6,
              "--class-weights", CLASS_W]
GRABCUT = {"mouse", "cup", "bottle", "toy_car"}
t_all = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])

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


def coco_root():
    for c in COCO_DIRS:
        for ann in list(c.glob("annotations/instances_train2017.json")) + list(c.glob("*/annotations/instances_train2017.json")):
            if (ann.parent.parent / "train2017").is_dir():
                return ann.parent.parent
    raise AssertionError(f"khong thay COCO trong {COCO_DIRS}")


TOOLS = find("tools_v3.txt").parent
COMP = find("classes.txt").parent
PROFILE = TOOLS / "ov7670_profile.json"
TMP, W = Path("/tmp/gen"), Path("/kaggle/working")
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
MAN = {"quick": QUICK, "steps": {}, "class_weights": CLASS_W, "paste_args": [str(x) for x in PASTE_ARGS]}


def step(name, t0):
    MAN["steps"][name] = round((time.time() - t0) / 60, 1)
    print(f"== {name}: {MAN['steps'][name]} phut", flush=True)


def parallel(cmds):
    ps = [subprocess.Popen([sys.executable, *map(str, c)]) for c in cmds]
    codes = [p.wait() for p in ps]
    assert all(c == 0 for c in codes), f"loi: {codes}"


def grabcut_one(path):
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
    if not 0.2 <= fg.mean() / 255 <= 0.95:
        return 0
    cv2.imwrite(path.rsplit(".", 1)[0] + "_m.png", fg)
    return 1


# ------------------------------------------------------------------ Roboflow: dung lai cua gen4 (khong dien nhan lai)
t0 = time.time()
for name in ("rf_train", "rf_val"):
    shutil.copy(find(f"gen4_{name}.tar"), W / f"gen5_{name}.tar")
pool = TMP / "pool.pkl"
shutil.copy(find("gen4_pool.pkl"), pool)
step("dung_lai_gen4", t0)

# ------------------------------------------------------------------ nen BTC (train 80%) cho ~40% anh khong dung giay
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
bt = TMP / "btc_train"
(bt / "images").mkdir(parents=True)
(bt / "labels").mkdir(parents=True)
for p in imgs[int(len(imgs) * 0.2):]:
    os.symlink(p, bt / "images" / p.name)
    lp = tr_lbl / (p.stem + ".txt")
    shutil.copy(lp, bt / "labels" / lp.name) if lp.exists() else (bt / "labels" / (p.stem + ".txt")).write_text("")

# ------------------------------------------------------------------ vat cat: Open Images + Hot Wheels (GrabCut) + TACO (mask san)
t0 = time.time()
crop_dirs = []
for tname, idx_name in (("oicrops.tar", "oicrops_index.csv"), ("tacocrops.tar", None)):
    hits = sorted(p for p in ALL_FILES if p.name == tname)
    if not hits:
        print(f"(khong co {tname})", flush=True)
        continue
    dst = TMP / tname.split(".")[0]
    with tarfile.open(hits[0]) as tar:
        tar.extractall(dst)
    if idx_name:
        shutil.copy(hits[0].parent / idx_name, dst / "crops" / "index.csv")
    crop_dirs.append(dst / "crops")
    print(tname, sum(1 for _ in csv.DictReader(open(dst / "crops" / "index.csv"))), "vat", flush=True)
hw_dir = find_dir("rf_hotwheels") or find_dir("edgeai-hotwheels")
if hw_dir is not None:
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
gc_files = [str(cd / r["file"]) for cd in crop_dirs if "taco" not in str(cd)
            for r in csv.DictReader(open(cd / "index.csv")) if r["cls"] in GRABCUT]
with Pool(NPROC) as pl:
    n_gc = sum(pl.map(grabcut_one, gc_files, chunksize=32))
MAN["grabcut"] = f"{n_gc}/{len(gc_files)}"
step("vat_cat_oi_hw_taco", t0)

# ------------------------------------------------------------------ dan (4 phan) -> mo phong
t0 = time.time()
croot = coco_root()
per = N_PASTE // NPROC
crop_args = [x for cd in crop_dirs for x in ("--crops", cd)]
parallel([[TOOLS / "coco_paste.py", "--images", croot, "--bg", bt, "--out", TMP / f"paste{k}", "--n", per,
           "--pool-cache", pool, "--prefix", f"g5p{k}", "--seed", 50 + k, *PASTE_ARGS, *crop_args]
          + (["--preview", W / "gen5_paste_preview.jpg"] if k == 0 else []) for k in range(NPROC)])
parallel([[TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / f"paste{k}", "--out",
           TMP / "coco" / f"p{k}", "--copies", 1, "--keep-orig", "--seed", 200 + k] for k in range(NPROC)])
step("dan_vat_va_mo_phong", t0)

t0 = time.time()
MAN["n_coco"] = sum(1 for _ in (TMP / "coco").rglob("images/*"))
with tarfile.open(W / "gen5_coco.tar", "w", dereference=True) as tar:
    tar.add(TMP / "coco", arcname="coco")
step("dong_goi", t0)
MAN["tong_phut"] = round((time.time() - t_all) / 60, 1)
(W / "gen5_manifest.json").write_text(json.dumps(MAN, indent=1))
print("XONG", json.dumps(MAN), flush=True)
