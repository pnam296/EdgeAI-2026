# gen3: TAO DU LIEU MOT LAN (chi CPU, khong ton quota GPU) cho cac ban train cam3 dung lai.
# Output (/kaggle/working): gen3_rf_train.tar, gen3_rf_val.tar, gen3_coco.tar, gen3_pool.pkl, gen3_manifest.json, anh xem truoc.
#   rf_train: Roboflow (rf_merged + Hot Wheels) da dien nhan thieu (YOLO11m COCO) -> zoom_out (vat du hinh) -> mo phong OV7670
#   rf_val  : 5% anh Roboflow goc giu rieng TRUOC khi ghep -> zoom -> mo phong
#   coco    : COCO 2017 + LVIS v1 cat theo mask, dan len nen BTC (80% train, SPLIT_SEED 42) -> anh goc + mo phong
# Moi buoc chia 4 phan chay song song (4 nhan CPU Kaggle).
import os, sys, subprocess, random, shutil, time, json, tarfile
from pathlib import Path

QUICK = os.environ.get("QUICK", "0") == "1"
N_COCO = 160 if QUICK else 4000
RF_RATIO = 0.03 if QUICK else 0.6
NPROC = 4
SPLIT_SEED = 42
t_all = time.time()

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199", "pycocotools"])

INP = Path("/kaggle/input")
ALL_FILES, ALL_DIRS, COCO_DIRS = [], [], []
for root, dirs, files in os.walk(INP, followlinks=True):
    keep = []
    for d in dirs:
        if "coco-2017" in d.lower():
            COCO_DIRS.append(Path(root) / d)
        else:
            keep.append(d)
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


TOOLS = find("tools_v3.txt").parent   # dataset edgeai-tools3 (ban tool moi nhat)
COMP = find("classes.txt").parent
CLASSES = [l.strip() for l in (COMP / "classes.txt").read_text().splitlines() if l.strip()]
PROFILE = TOOLS / "ov7670_profile.json"
TMP, W = Path("/tmp/gen"), Path("/kaggle/working")
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
(TMP / "classes.txt").write_text("\n".join(CLASSES) + "\n")
MAN = {"quick": QUICK, "steps": {}}


def step(name, t0):
    MAN["steps"][name] = round((time.time() - t0) / 60, 1)
    print(f"== {name}: {MAN['steps'][name]} phut", flush=True)


def mkset(d, items):
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    for ip, lp in items:
        os.symlink(ip, d / "images" / ip.name)
        dst = d / "labels" / (ip.stem + ".txt")
        if lp.exists():
            shutil.copy(lp, dst)
        else:
            dst.write_text("")
    return d


def parallel(cmds):
    """chay song song cac lenh python, loi o bat ky lenh nao -> dung"""
    ps = [subprocess.Popen([sys.executable, *map(str, c)]) for c in cmds]
    codes = [p.wait() for p in ps]
    assert all(c == 0 for c in codes), f"loi: {codes}"


# ------------------------------------------------------------------ nen BTC (train 80%, giong train_cam3)
t0 = time.time()
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
btc_train = mkset(TMP / "btc_train", [(p, tr_lbl / (p.stem + ".txt")) for p in imgs[int(len(imgs) * 0.2):]])

# ------------------------------------------------------------------ Roboflow: dien nhan thieu (CPU, YOLO11m)
rf_items = []
for rname in ("rf_merged", "rf_hotwheels", "edgeai-hotwheels"):
    d = find_dir(rname)
    if d is None:
        print(f"(khong co {rname})", flush=True)
        continue
    its = [(p, d / "labels" / (p.stem + ".txt")) for p in sorted((d / "images").iterdir())]
    print(f"{rname}: {len(its)} anh", flush=True)
    MAN[f"src_{rname}"] = len(its)
    rf_items += its
random.Random(1).shuffle(rf_items)
if QUICK:
    rf_items = rf_items[:400]
n_rv = max(1, len(rf_items) // 20)
parts = {"rfv": rf_items[:n_rv], "rft": rf_items[n_rv:]}
COCO_IDS = {39: 0, 40: 1, 41: 1, 67: 2, 73: 3, 64: 4}
import torch
torch.set_num_threads(NPROC)
from ultralytics import YOLO
filler = YOLO("yolo11m.pt")


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
    for k in range(0, len(ims), 32):
        for r in filler.predict([str(p) for p in ims[k:k + 32]], imgsz=480, conf=0.5, classes=list(COCO_IDS),
                                verbose=False, device="cpu"):
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
        if k % 1600 == 0:
            print(f"  dien nhan {part}: {k}/{len(ims)}", flush=True)
MAN["rf_filled_boxes"] = n_fill
del filler
step("dien_nhan_roboflow", t0)

# ------------------------------------------------------------------ zoom (4 phan song song) -> mo phong
t0 = time.time()
rft = sorted((TMP / "rft_filled" / "images").iterdir())
shards = []
for k in range(NPROC):
    its = [(p.resolve(), TMP / "rft_filled" / "labels" / (p.stem + ".txt")) for p in rft[k::NPROC]]
    shards.append(mkset(TMP / f"rft_in{k}", its))
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

# ------------------------------------------------------------------ COCO + LVIS: loc kho vat 1 lan -> dan (4 phan) -> mo phong
t0 = time.time()
ann, croot = find_coco()
lvis = []
for part in ("train", "val"):
    z = TMP / f"lvis_v1_{part}.json.zip"
    subprocess.check_call(["wget", "-q", "-O", str(z), f"https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{part}.json.zip"])
    subprocess.check_call(["unzip", "-q", "-o", str(z), "-d", str(TMP)])
    z.unlink()
    lvis.append(TMP / f"lvis_v1_{part}.json")
pool = W / "gen3_pool.pkl"
subprocess.check_call([sys.executable, str(TOOLS / "coco_paste.py"), "--ann", str(ann),
                       *[x for p in lvis for x in ("--ann", str(p))], "--images", str(croot), "--bg", str(btc_train),
                       "--out", str(TMP / "unused"), "--pool-cache", str(pool), "--only-pool"])
for p in lvis:
    p.unlink()
step("loc_kho_vat_coco_lvis", t0)
t0 = time.time()
per = N_COCO // NPROC
parallel([[TOOLS / "coco_paste.py", "--images", croot, "--bg", btc_train, "--out", TMP / f"coco_paste{k}",
           "--n", per, "--pool-cache", pool, "--prefix", f"coco{k}", "--seed", k]
          + (["--preview", W / "coco_paste_preview.jpg"] if k == 0 else []) for k in range(NPROC)])
parallel([[TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / f"coco_paste{k}", "--out",
           TMP / "coco" / f"p{k}", "--copies", 1, "--keep-orig", "--seed", 100 + k] for k in range(NPROC)])
step("dan_coco_lvis_va_mo_phong", t0)

# ------------------------------------------------------------------ dong goi (1 file/phan, khong de Kaggle luu hang chuc nghin file le)
t0 = time.time()
for name in ("rf_train", "rf_val", "coco"):
    n = sum(1 for _ in (TMP / name).rglob("images/*"))
    MAN[f"n_{name}"] = n
    with tarfile.open(W / f"gen3_{name}.tar", "w", dereference=True) as tar:   # luu FILE that, khong luu symlink
        tar.add(TMP / name, arcname=name, recursive=True, filter=None)
    print(f"{name}: {n} anh -> gen3_{name}.tar ({(W / f'gen3_{name}.tar').stat().st_size / 1e6:.0f} MB)", flush=True)
step("dong_goi", t0)
MAN["tong_phut"] = round((time.time() - t_all) / 60, 1)
(W / "gen3_manifest.json").write_text(json.dumps(MAN, indent=1))
print("XONG", json.dumps(MAN), flush=True)
