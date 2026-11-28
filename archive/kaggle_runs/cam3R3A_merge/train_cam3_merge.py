"""Train Cam3R3A merged: gen5 (Roboflow + HotWheels + COCO/LVIS + OpenImages + TACO + BTC sim)
+ custom camera dataset (1075 cam_real train x REAL_REP, 215 cam_val val).

Account: dohuuminhtri
Kernel : dohuuminhtri/edgeai-cam3r3a-merge
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

QUICK = os.environ.get("QUICK", "0") == "1"
T_EPOCHS = 2 if QUICK else 15
S_EPOCHS = 2 if QUICK else 45
T_LR = 0.002
S_LR = 0.002
REAL_REP = 3          # 1075 x 3 = 3225 cam_real views per epoch alongside ~15k gen5 images
BATCH = 64
DIST_SCALE = (1.0, 1.0, 1.5)
KD_CLS = KD_BOX = 1.0
SPLIT_SEED = 42
AUG = dict(hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.5, fliplr=0.5, flipud=0.5, close_mosaic=10)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199"])

INP = Path("/kaggle/input")
ALL_FILES, ALL_DIRS = [], []
for root, dirs, files in os.walk(INP, followlinks=True):
    keep = [d for d in dirs if "coco-2017" not in d.lower()]
    dirs[:] = keep
    ALL_DIRS += [Path(root) / d for d in keep]
    ALL_FILES += [Path(root) / f for f in files]
print(f"/kaggle/input: {len(ALL_FILES)} files", flush=True)


def find(name: str, must_contain: str = "") -> Path:
    hits = sorted(
        p for p in ALL_FILES if p.name == name and (not must_contain or must_contain in p.as_posix().lower())
    )
    assert hits, f"khong thay {name} (filter={must_contain!r}) trong /kaggle/input"
    return hits[0]


def find_dir(name: str) -> Path:
    hits = sorted(p for p in ALL_DIRS if p.name == name and (p / "images").is_dir())
    assert hits, f"khong thay thu muc {name}/images"
    return hits[0]


def newest(pattern: str) -> Path:
    hits = sorted(p for p in ALL_FILES if re.fullmatch(pattern, p.name))
    assert hits, f"khong thay trong so {pattern}"
    print("dung trong so:", hits[0], flush=True)
    return hits[0]


# Locate competition dir (has train/images + classes.txt), tools dir, and custom cam dataset
COMP_HITS = sorted(p.parent for p in ALL_FILES if p.name == "classes.txt" and (p.parent / "train" / "images").is_dir())
assert COMP_HITS, "khong thay thu muc cuoc thi (co classes.txt + train/images)"
COMP = COMP_HITS[0]

TOOLS = find("yolo_reach.py").parent
sys.path.insert(0, str(TOOLS))
import yolo_reach

print("tam voi:", yolo_reach.apply(DIST_SCALE), "| box toi da:", yolo_reach.max_box(), flush=True)

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

W = Path("/kaggle/working")
TMP = Path("/tmp/gen")
DS = Path("/tmp/ds")
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(DS, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)
DS.mkdir(parents=True, exist_ok=True)

CLASSES = [l.strip() for l in (COMP / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
names = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]"


def mkset(d: Path, items):
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    for ip, lp in items:
        os.symlink(ip, d / "images" / ip.name)
        if lp.exists():
            shutil.copy(lp, d / "labels" / (ip.stem + ".txt"))
        else:
            (d / "labels" / (ip.stem + ".txt")).write_text("")
    return d


t0 = time.time()
# ------------------------------------------------------------------ 1) BTC sim + BTC val
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
n_val = max(1, int(len(imgs) * 0.2))
val_btc, train_btc = imgs[:n_val], imgs[n_val:]
btc_val = mkset(TMP / "btc_val", [(p, tr_lbl / (p.stem + ".txt")) for p in val_btc])

tsim = find_dir("train_sim")
stems = {p.stem for p in train_btc}
btc_sim = mkset(
    TMP / "btc_sim",
    [(p, tsim / "labels" / (p.stem + ".txt")) for p in sorted((tsim / "images").iterdir()) if p.stem.rsplit("_sim", 1)[0] in stems],
)

# ------------------------------------------------------------------ 2) Unpack gen5 (Roboflow + HotWheels + COCO/LVIS + OI + TACO)
GEN_MAN = sorted(
    (p for p in ALL_FILES if re.fullmatch(r"gen\d+_manifest\.json", p.name)),
    key=lambda p: int(re.findall(r"\d+", p.name)[0]),
    reverse=True,
)
assert GEN_MAN, "khong thay gen5_manifest.json tu edgeai-gen5"
print(f"Unpacking {GEN_MAN[0].name}:", GEN_MAN[0].read_text()[:400], flush=True)
for part in ("rf_train", "rf_val", "coco"):
    tpath = GEN_MAN[0].parent / GEN_MAN[0].name.replace("manifest.json", f"{part}.tar")
    with tarfile.open(tpath) as tar:
        tar.extractall(TMP)
RF_TRAIN = sorted((TMP / "rf_train").glob("p*"))
RF_VAL = sorted((TMP / "rf_val").glob("p*"))
COCO_OUT = sorted((TMP / "coco").glob("p*"))

# ------------------------------------------------------------------ 3) Unpack custom cam_real (1075) + cam_val (215)
custom_roots = sorted(
    p for p in ALL_DIRS if "edgeai-cam-custom-ds" in p.as_posix().lower() and p.name == "edgeai-cam-custom-ds"
)
if not custom_roots:
    custom_roots = [find("classes.txt", "edgeai-cam-custom-ds").parent]
CUSTOM = custom_roots[0]
print("CUSTOM root:", CUSTOM, "| contents:", [p.name for p in CUSTOM.iterdir()], flush=True)


def prepare_custom_split(split_name: str) -> Path:
    direct = CUSTOM / split_name
    if (direct / "images").is_dir() and (direct / "labels").is_dir():
        return direct
    zf = CUSTOM / f"{split_name}.zip"
    if zf.is_file():
        dst = TMP / f"custom_{split_name}"
        with zipfile.ZipFile(zf) as z:
            z.extractall(dst)
        if (dst / split_name / "images").is_dir():
            return dst / split_name
        if (dst / "images").is_dir():
            return dst
    raise FileNotFoundError(f"Khong tim thay {split_name}/images trong {CUSTOM}")


CAM_TRAIN = prepare_custom_split("cam_real")
CAM_VAL = prepare_custom_split("cam_val")
print(f"Unpack xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)


# ------------------------------------------------------------------ 4) Merge all into /tmp/ds
def link_all(src: Path, split: str, prefix: str) -> int:
    n = 0
    (DS / "images" / split).mkdir(parents=True, exist_ok=True)
    (DS / "labels" / split).mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "images").iterdir()):
        if not p.is_file():
            continue
        lp = src / "labels" / (p.stem + ".txt")
        os.symlink(p.resolve(), DS / "images" / split / f"{prefix}_{p.name}")
        dst = DS / "labels" / split / f"{prefix}_{p.stem}.txt"
        if lp.exists():
            os.symlink(lp.resolve(), dst)
        else:
            dst.write_text("")
        n += 1
    return n


cnt = {
    "btc_sim": link_all(btc_sim, "train", "btcsim"),
    "rf": sum(link_all(d, "train", f"rf{k}") for k, d in enumerate(RF_TRAIN)),
    "coco": sum(link_all(d, "train", f"coco{k}") for k, d in enumerate(COCO_OUT)),
    "cam_real_xrep": sum(link_all(CAM_TRAIN, "train", f"camr{r}") for r in range(REAL_REP)),
    "val_btc": link_all(btc_val, "val", "btc"),
    "val_rf": sum(link_all(d, "rfval", f"rf{k}") for k, d in enumerate(RF_VAL)),
    "val_cam": link_all(CAM_VAL, "camval", "cv"),
}
real55 = find("split.json").parent
cnt["test_real55"] = link_all(real55 / "train", "real", "a") + link_all(real55 / "test", "real", "b")
cnt["test_btcmau"] = link_all(find_dir("btc_samples"), "btcmau", "m")
print("Dataset counts:", cnt, flush=True)


def yaml_for(val):
    v = "[" + ", ".join(val) + "]" if isinstance(val, list) else val
    return f"path: {DS}\ntrain: images/train\nval: {v}\nnames: {names}\n"


# Select best checkpoint across BTC val + Roboflow val + custom cam_val
(DS / "data.yaml").write_text(yaml_for(["images/val", "images/rfval", "images/camval"]), encoding="utf-8")
for k, v in (
    ("btc", "images/val"),
    ("rf", "images/rfval"),
    ("camval", "images/camval"),
    ("real", "images/real"),
    ("btcmau", "images/btcmau"),
):
    (DS / f"{k}.yaml").write_text(yaml_for(v), encoding="utf-8")


def evaluate(weights: Path, tag: str) -> dict:
    out = {}
    for split in ("camval", "btc", "rf", "real", "btcmau"):
        r = YOLO(str(weights)).val(
            data=str(DS / f"{split}.yaml"),
            imgsz=320,
            batch=64,
            device=0,
            plots=False,
            verbose=False,
            project="/tmp/val",
            name=f"{tag}_{split}",
            exist_ok=True,
        )
        out[split] = {
            "map50": round(float(r.box.map50), 4),
            "per_class": {CLASSES[int(c)]: round(float(a), 3) for c, a in zip(r.box.ap_class_index, r.box.ap50)},
        }
        print(f"[{tag}] {split:6} mAP50={out[split]['map50']} {out[split]['per_class']}", flush=True)
    return out


def load_matching(model, sd):
    own = model.state_dict()
    take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    return len(take), len(own)


RES = {
    "variant": "cam3R3A_merge",
    "counts": cnt,
    "real_rep": REAL_REP,
    "t_epochs": T_EPOCHS,
    "s_epochs": S_EPOCHS,
    "dist_scale": list(DIST_SCALE),
}
TEACHER_YAML_S = (TOOLS / "teacher_p2p4s.yaml").read_text().replace("\r", "")

# ------------------------------------------------------------------ 5) TEACHER (from cam3C)
t0 = time.time()
tcfg = Path("/tmp/teacher_p2p4s.yaml")
tcfg.write_text(TEACHER_YAML_S, encoding="utf-8")
teacher_init = newest(r"yolo_cam3[A-D]_teacher\.pt")
teacher = YOLO(str(tcfg)).load(str(teacher_init))
print(f"Teacher: {sum(p.numel() for p in teacher.model.parameters()):,} params", flush=True)
teacher.train(
    data=str(DS / "data.yaml"),
    epochs=T_EPOCHS,
    imgsz=320,
    batch=BATCH,
    workers=4,
    device=0,
    pretrained=True,
    optimizer="SGD",
    lr0=T_LR,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=1,
    warmup_bias_lr=T_LR,
    project=str(W),
    name="teacher",
    exist_ok=True,
    plots=False,
    patience=0,
    seed=0,
    **AUG,
)
TEACHER_W = W / "teacher" / "weights" / "last.pt"
print(f"Teacher done in {(time.time() - t0) / 60:.1f} min", flush=True)
RES["teacher"] = evaluate(TEACHER_W, "teacher")


# ------------------------------------------------------------------ 6) STUDENT P2P4 + KD (from cam3C)
class KDLoss(v8DetectionLoss):
    def __init__(self, model, teacher_m):
        super().__init__(model)
        self.teacher = teacher_m

    def loss(self, preds, batch):
        total, items = super().loss(preds, batch)
        with torch.no_grad():
            tp = self.parse_output(self.teacher(batch["img"]))
        B = preds["scores"].shape[0]
        s_cls, t_cls = preds["scores"].float(), tp["scores"].float()
        t_p = t_cls.sigmoid()
        norm = t_p.sum().clamp(min=1)
        kd_cls = F.binary_cross_entropy_with_logits(s_cls, t_p, reduction="sum") / norm
        R = self.reg_max
        s_box = preds["boxes"].float().view(B, 4, R, -1)
        t_box = tp["boxes"].float().view(B, 4, R, -1)
        kl = (F.softmax(t_box, 2) * (F.log_softmax(t_box, 2) - F.log_softmax(s_box, 2))).sum(2)
        kd_box = (kl.sum(1) * t_p.max(1).values).sum() / norm
        return total + (KD_CLS * kd_cls + KD_BOX * kd_box) * B, items


def attach_kd(trainer):
    t = YOLO(str(TEACHER_W)).model.to(trainer.device).float().eval()
    for p in t.parameters():
        p.requires_grad_(False)
    t_out = t(torch.zeros(1, 3, 240, 320, device=trainer.device))
    s_out = trainer.model.eval()(torch.zeros(1, 3, 240, 320, device=trainer.device))
    trainer.model.train()
    assert t_out[1]["boxes"].shape == s_out[1]["boxes"].shape
    trainer.model.criterion = KDLoss(trainer.model, t)
    print(f"Distill from {TEACHER_W.name} ({sum(p.numel() for p in t.parameters()):,} params)", flush=True)


t0 = time.time()
student = YOLO(str(find("student_p2p4.yaml")))
student_init = newest(r"yolo_cam3[A-D]_last\.pt")
init_sd = YOLO(str(student_init)).model.state_dict()
print("Student: loaded %d/%d tensors from cam3C" % load_matching(student.model, init_sd), flush=True)
student.load(student.model)
n_params = sum(p.numel() for p in student.model.parameters())
print(f"Student P2P4: {n_params:,} params", flush=True)
assert n_params <= 200_000

student.add_callback("on_train_start", attach_kd)
student.train(
    data=str(DS / "data.yaml"),
    epochs=S_EPOCHS,
    imgsz=320,
    batch=BATCH,
    workers=4,
    device=0,
    pretrained=True,
    optimizer="SGD",
    lr0=S_LR,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=1,
    warmup_bias_lr=S_LR,
    project=str(W),
    name="student",
    exist_ok=True,
    plots=False,
    patience=0,
    seed=0,
    **AUG,
)
print(f"Student done in {(time.time() - t0) / 60:.1f} min", flush=True)

for wn in ("best.pt", "last.pt"):
    RES["student_" + wn] = evaluate(W / "student" / "weights" / wn, "student_" + wn)
RES["cam3_init"] = evaluate(student_init, "cam3_init")

(W / "results_cam.json").write_text(json.dumps(RES, indent=2), encoding="utf-8")
for src, dst in (
    ("student/weights/best.pt", "yolo_cam3R3A_merge_best.pt"),
    ("student/weights/last.pt", "yolo_cam3R3A_merge_last.pt"),
    ("teacher/weights/last.pt", "yolo_cam3R3A_merge_teacher.pt"),
):
    shutil.copy(W / src, W / dst)
shutil.rmtree(W / "teacher", ignore_errors=True)
shutil.rmtree(W / "student", ignore_errors=True)
summary = {
    k: {s: v[s]["map50"] for s in v}
    for k, v in RES.items()
    if isinstance(v, dict) and "camval" in v
}
print("XONG cam3R3A_merge!", json.dumps(summary), flush=True)
