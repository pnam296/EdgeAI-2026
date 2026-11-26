# cam2: fine-tune cam1 bang anh CAMERA THAT (tap trung lop mouse: student cam1 gan nhu khong nhan chuot that).
#  1) THAY: tu yolo_cam_teacher.pt (cam1), train tiep tren du lieu cu + anh that x REAL_REP + anh dan chuot that
#  2) TRO: tu yolo_cam_best.pt (cam1), distill tu thay moi (KDLoss)
# cls_pw: trong so lop theo nghich tan suat (lop it vi du nhu mouse duoc nang).
# Val chon best: BTC val + anh that test (13 anh, tach theo canh, khong lap voi train).
import os, sys, subprocess, random, shutil, time, json
from pathlib import Path

T_EPOCHS = int(os.environ.get("T_EPOCHS", 25))
S_EPOCHS = int(os.environ.get("S_EPOCHS", 50))
REAL_REP = 15        # anh that it -> lap lai trong tap train
CLS_PW = 0.5
QUICK = os.environ.get("QUICK", "0") == "1"           # 1: 2 epoch moi giai doan, thu pipeline
BATCH = 64
DIST_SCALE = (1.0, 1.0, 1.5)
KD_CLS, KD_BOX = 1.0, 1.0
SPLIT_SEED = 42
AUG = dict(hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.4, fliplr=0.5, close_mosaic=10)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199"])

INP = Path("/kaggle/input")
def find(name):
    hits = sorted(INP.rglob(name))
    assert hits, f"khong thay {name} trong /kaggle/input"
    return hits[0]

def find_dir(name):
    hits = sorted(p for p in INP.rglob(name) if p.is_dir() and (p / "images").is_dir())
    assert hits, f"khong thay thu muc {name}/images trong /kaggle/input"
    return hits[0]

CAM = find("yolo_reach.py").parent
COMP = find("classes.txt").parent
sys.path.insert(0, str(CAM))
import yolo_reach
print("tam voi:", yolo_reach.apply(DIST_SCALE), "| box toi da:", yolo_reach.max_box(), flush=True)

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

W = Path("/kaggle/working")
DS = Path("/tmp/ds")
CLASSES = [l.strip() for l in (COMP / "classes.txt").read_text().splitlines() if l.strip()]
names = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]"

TEACHER_YAML = """nc: 7
reg_max: 4
scales:
  s: [0.5, 0.5, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, SPPF, [512, 5]]
  - [-1, 2, C2PSA, [512]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]
  - [[14, 17, 20], 1, Detect, [nc]]
"""
TEACHER_CFG = Path("/tmp/teacher_p2p4s.yaml")   # ten co chu 's' -> Ultralytics chon scale s
TEACHER_CFG.write_text(TEACHER_YAML)


# ------------------------------------------------------------------ dataset
def link(src_img, src_lbl, split, name):
    (DS / "images" / split).mkdir(parents=True, exist_ok=True)
    (DS / "labels" / split).mkdir(parents=True, exist_ok=True)
    os.symlink(src_img, DS / "images" / split / (name + src_img.suffix))
    dst = DS / "labels" / split / (name + ".txt")
    if src_lbl.exists():
        os.symlink(src_lbl, dst)
    else:
        dst.write_text("")


shutil.rmtree(DS, ignore_errors=True)
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
n_val = max(1, int(len(imgs) * 0.2))
val_btc, train_btc = imgs[:n_val], imgs[n_val:]
for p in train_btc:
    link(p, tr_lbl / (p.stem + ".txt"), "train", "btc_" + p.stem)
for p in val_btc:
    link(p, tr_lbl / (p.stem + ".txt"), "val", "btc_" + p.stem)

tsim = find_dir("train_sim")
train_stems = {p.stem for p in train_btc}
n_tsim = 0
for p in sorted((tsim / "images").iterdir()):
    if p.stem.rsplit("_sim", 1)[0] in train_stems:   # chi ban mo phong cua anh train (khong ro ri val)
        link(p, tsim / "labels" / (p.stem + ".txt"), "train", "btcsim_" + p.stem)
        n_tsim += 1

rsim = find_dir("rf_sim")
rimgs = sorted((rsim / "images").iterdir())
bases = sorted({p.stem.rsplit("_sim", 1)[0] for p in rimgs})
random.Random(1).shuffle(bases)
rf_val = set(bases[:max(1, len(bases) // 20)])
n_rf = [0, 0]
for p in rimgs:
    b = p.stem.rsplit("_sim", 1)[0]
    is_val = b in rf_val
    if is_val and "_sim" not in p.stem:
        continue
    link(p, rsim / "labels" / (p.stem + ".txt"), "rfval" if is_val else "train", "rf_" + p.stem)
    n_rf[is_val] += 1
real = find("split.json").parent      # cam_real/{train,test}/{images,labels}
n_real = 0
for r in range(REAL_REP):
    for p in sorted((real / "train" / "images").iterdir()):
        link(p, real / "train" / "labels" / (p.stem + ".txt"), "train", f"real{r}_" + p.stem)
        n_real += 1
for p in sorted((real / "test" / "images").iterdir()):
    link(p, real / "test" / "labels" / (p.stem + ".txt"), "realtest", "real_" + p.stem)
pm = find_dir("paste_mouse")
n_paste = 0
for p in sorted((pm / "images").iterdir()):
    link(p, pm / "labels" / (p.stem + ".txt"), "train", p.stem)
    n_paste += 1
print(f"them: anh that {n_real} (x{REAL_REP}), dan chuot {n_paste}, test that "
      f"{len(list((real / 'test' / 'images').iterdir()))}", flush=True)
if QUICK:
    for split, keep in (("train", 256), ("val", 64), ("rfval", 64)):
        for k, p in enumerate(sorted((DS / "images" / split).iterdir())):
            if k >= keep:
                p.unlink()
                (DS / "labels" / split / (p.stem + ".txt")).unlink()
print(f"train: BTC {len(train_btc)} + BTC sim {n_tsim} + RF {n_rf[0]} | val BTC {len(val_btc)} | val RF {n_rf[1]}",
      flush=True)

def yaml_for(val):
    v = "[" + ", ".join(val) + "]" if isinstance(val, list) else val
    return f"path: {DS}\ntrain: images/train\nval: {v}\nnames: {names}\n"

(DS / "data.yaml").write_text(yaml_for(["images/val", "images/realtest"]))   # chon best: BTC + anh that
(DS / "btc.yaml").write_text(yaml_for("images/val"))
(DS / "rf.yaml").write_text(yaml_for("images/rfval"))
(DS / "real.yaml").write_text(yaml_for("images/realtest"))


def evaluate(weights, tag):
    out = {}
    for split in ("btc", "rf", "real"):
        r = YOLO(str(weights)).val(data=str(DS / f"{split}.yaml"), imgsz=320, batch=64, device=0, plots=False,
                                   verbose=False, project="/tmp/val", name=f"{tag}_{split}", exist_ok=True)
        out[split] = {"map50": round(float(r.box.map50), 4),
                      "per_class": {CLASSES[int(c)]: round(float(a), 3)   # theo ap_class_index: tap that khong co box
                                    for c, a in zip(r.box.ap_class_index, r.box.ap50)}}
        print(f"[{tag}] {split} mAP50 {out[split]['map50']} {out[split]['per_class']}", flush=True)
    return out


RES = {}
# ------------------------------------------------------------------ 1) THAY
t0 = time.time()
teacher = YOLO(str(TEACHER_CFG)).load(str(find("yolo_cam_teacher.pt")))
print(f"thay: {sum(p.numel() for p in teacher.model.parameters()):,} tham so, tu cam1", flush=True)
teacher.train(data=str(DS / "data.yaml"), epochs=2 if QUICK else T_EPOCHS, imgsz=320, batch=BATCH, workers=4,
              device=0, pretrained=True, optimizer="SGD", lr0=0.002, lrf=0.05, cos_lr=True, warmup_epochs=1,
              warmup_bias_lr=0.002, cls_pw=CLS_PW,
              project=str(W), name="teacher", exist_ok=True, plots=False, patience=0, seed=0, **AUG)
TEACHER_W = W / "teacher" / "weights" / "best.pt"
print(f"thay xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)
RES["teacher"] = evaluate(TEACHER_W, "thay")


# ------------------------------------------------------------------ 2) TRO + distill
class KDLoss(v8DetectionLoss):
    def __init__(self, model, teacher):
        super().__init__(model)
        self.teacher = teacher

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
    assert t_out[1]["boxes"].shape == s_out[1]["boxes"].shape, "Thay va tro khac bo cuc dau ra"
    trainer.model.criterion = KDLoss(trainer.model, t)
    print(f"distill tu {TEACHER_W.name} ({sum(p.numel() for p in t.parameters()):,} tham so)", flush=True)


t0 = time.time()
student = YOLO(str(find("student_p2p4.yaml"))).load(str(find("yolo_cam_best.pt")))
n = sum(p.numel() for p in student.model.parameters())
print(f"tro: {n:,} tham so", flush=True)
assert n <= 200_000
student.add_callback("on_train_start", attach_kd)
student.train(data=str(DS / "data.yaml"), epochs=2 if QUICK else S_EPOCHS, imgsz=320, batch=BATCH, workers=4,
              device=0, pretrained=True, optimizer="SGD", lr0=0.002, lrf=0.05, cos_lr=True, warmup_epochs=1, warmup_bias_lr=0.002, cls_pw=CLS_PW,
              project=str(W), name="student", exist_ok=True, plots=False, patience=0, seed=0, **AUG)
print(f"tro xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)
for wn in ("best.pt", "last.pt"):
    RES["student_" + wn] = evaluate(W / "student" / "weights" / wn, "tro_" + wn)

RES["cam1_student"] = evaluate(find("yolo_cam_best.pt"), "cam1")

(W / "results_cam.json").write_text(json.dumps(RES, indent=1))
for src, dst in (("student/weights/best.pt", "yolo_cam2_best.pt"), ("student/weights/last.pt", "yolo_cam2_last.pt"),
                 ("teacher/weights/best.pt", "yolo_cam2_teacher.pt")):
    shutil.copy(W / src, W / dst)
print("XONG", json.dumps({k: {s: v[s]["map50"] for s in v} for k, v in RES.items()}), flush=True)
