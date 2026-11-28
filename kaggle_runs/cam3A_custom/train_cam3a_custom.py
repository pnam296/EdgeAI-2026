"""Train Cam3A (Teacher P2P4 + Student P2P4 with KD) on custom cam_real (1075) / cam_val (215) dataset.

Account: dohuuminhtri
Inputs:
  - dohuuminhtri/edgeai-cam3-all      (yolo_reach.py, teacher_p2p4s.yaml, student_p2p4.yaml, yolo_cam_teacher.pt, yolo_cam_best.pt)
  - dohuuminhtri/edgeai-cam-custom-ds (cam_real/{images,labels}, cam_val/{images,labels}, classes.txt)
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

QUICK = os.environ.get("QUICK", "0") == "1"
T_EPOCHS = 2 if QUICK else 30
S_EPOCHS = 2 if QUICK else 60
T_LR = 0.005
S_LR = 0.002
BATCH = 64
DIST_SCALE = (1.0, 1.0, 1.5)
KD_CLS, KD_BOX = 1.0, 1.0
AUG = dict(hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.5, fliplr=0.5, flipud=0.5, close_mosaic=10)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199"])

INP = Path("/kaggle/input")
ALL_FILES, ALL_DIRS = [], []
for root, dirs, files in os.walk(INP, followlinks=True):
    ALL_DIRS += [Path(root) / d for d in dirs]
    ALL_FILES += [Path(root) / f for f in files]
print(f"/kaggle/input: {len(ALL_FILES)} files", flush=True)


def find(name: str, must_contain: str = "") -> Path:
    hits = sorted(
        p for p in ALL_FILES if p.name == name and (not must_contain or must_contain in p.as_posix().lower())
    )
    assert hits, f"khong thay {name} (filter={must_contain!r}) trong /kaggle/input"
    return hits[0]


# Locate tools & init weights from edgeai-cam3-all
TOOLS = find("yolo_reach.py").parent
sys.path.insert(0, str(TOOLS))
import yolo_reach

print("tam voi:", yolo_reach.apply(DIST_SCALE), "| box toi da:", yolo_reach.max_box(), flush=True)

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

W = Path("/kaggle/working")
DS = Path("/tmp/cam_ds")
shutil.rmtree(DS, ignore_errors=True)
DS.mkdir(parents=True, exist_ok=True)

# Locate custom dataset (handle either extracted folders or zip files from --dir-mode zip)
custom_roots = sorted(
    p for p in ALL_DIRS if "edgeai-cam-custom-ds" in p.as_posix().lower() and p.name == "edgeai-cam-custom-ds"
)
if not custom_roots:
    custom_roots = [find("classes.txt", "edgeai-cam-custom-ds").parent]
CUSTOM = custom_roots[0]
print("CUSTOM root:", CUSTOM, "| contents:", [p.name for p in CUSTOM.iterdir()], flush=True)


def prepare_split(split_name: str) -> Path:
    direct = CUSTOM / split_name
    if (direct / "images").is_dir() and (direct / "labels").is_dir():
        return direct
    zf = CUSTOM / f"{split_name}.zip"
    if zf.is_file():
        dst = DS / split_name
        with zipfile.ZipFile(zf) as z:
            z.extractall(dst)
        # Handle nested split_name/split_name/images if present
        if (dst / split_name / "images").is_dir():
            return dst / split_name
        if (dst / "images").is_dir():
            return dst
    raise FileNotFoundError(f"Khong tim thay {split_name}/images trong {CUSTOM}")


TRAIN_DIR = prepare_split("cam_real")
VAL_DIR = prepare_split("cam_val")
n_tr = len(list((TRAIN_DIR / "images").iterdir()))
n_va = len(list((VAL_DIR / "images").iterdir()))
print(f"Dataset ready | train: {TRAIN_DIR} ({n_tr} imgs) | val: {VAL_DIR} ({n_va} imgs)", flush=True)

CLASSES = [l.strip() for l in (CUSTOM / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
names_str = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]"
DATA_YAML = DS / "data.yaml"
DATA_YAML.write_text(
    f"train: {TRAIN_DIR / 'images'}\nval: {VAL_DIR / 'images'}\nnc: {len(CLASSES)}\nnames: {names_str}\n",
    encoding="utf-8",
)


def evaluate(weights: Path, tag: str) -> dict:
    r = YOLO(str(weights)).val(
        data=str(DATA_YAML),
        imgsz=320,
        batch=64,
        device=0,
        plots=False,
        verbose=False,
        project="/tmp/val",
        name=tag,
        exist_ok=True,
    )
    out = {
        "map50": round(float(r.box.map50), 4),
        "map50_95": round(float(r.box.map), 4),
        "per_class": {CLASSES[int(c)]: round(float(a), 3) for c, a in zip(r.box.ap_class_index, r.box.ap50)},
    }
    print(f"[{tag}] cam_val mAP50={out['map50']} mAP50-95={out['map50_95']} {out['per_class']}", flush=True)
    return out


def load_matching(model, sd):
    own = model.state_dict()
    take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    return len(take), len(own)


RES = {
    "variant": "cam3A_custom",
    "train_imgs": n_tr,
    "val_imgs": n_va,
    "t_epochs": T_EPOCHS,
    "s_epochs": S_EPOCHS,
    "dist_scale": list(DIST_SCALE),
}

# Baseline before fine-tuning
CAM1_BEST = find("yolo_cam_best.pt")
RES["cam1_baseline"] = evaluate(CAM1_BEST, "cam1_baseline")

# ------------------------------------------------------------------ 1) TEACHER
t0 = time.time()
tcfg = Path("/tmp/teacher_p2p4s.yaml")
tcfg.write_text(find("teacher_p2p4s.yaml").read_text().replace("\r", ""))
teacher = YOLO(str(tcfg)).load(str(find("yolo_cam_teacher.pt")))
print(f"Teacher: {sum(p.numel() for p in teacher.model.parameters()):,} params", flush=True)
teacher.train(
    data=str(DATA_YAML),
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
RES["teacher_best"] = evaluate(W / "teacher" / "weights" / "best.pt", "teacher_best")
RES["teacher_last"] = evaluate(TEACHER_W, "teacher_last")


# ------------------------------------------------------------------ 2) STUDENT + KD
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
cam1_sd = YOLO(str(CAM1_BEST)).model.state_dict()
print("Student: loaded %d/%d tensors from cam1" % load_matching(student.model, cam1_sd), flush=True)
student.load(student.model)
n_params = sum(p.numel() for p in student.model.parameters())
print(f"Student P2P4: {n_params:,} params", flush=True)
assert n_params <= 200_000

student.add_callback("on_train_start", attach_kd)
student.train(
    data=str(DATA_YAML),
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

(W / "results_cam.json").write_text(json.dumps(RES, indent=2), encoding="utf-8")
for src, dst in (
    ("student/weights/best.pt", "yolo_cam3A_custom_best.pt"),
    ("student/weights/last.pt", "yolo_cam3A_custom_last.pt"),
    ("teacher/weights/last.pt", "yolo_cam3A_custom_teacher.pt"),
):
    shutil.copy(W / src, W / dst)
shutil.rmtree(W / "teacher", ignore_errors=True)
shutil.rmtree(W / "student", ignore_errors=True)
print("XONG cam3A_custom!", json.dumps({k: v["map50"] for k, v in RES.items() if isinstance(v, dict) and "map50" in v}), flush=True)
