"""cam4: thầy + trò train lại trên gen5 + ảnh BTC mô phỏng + ẢNH CAMERA THẬT (khăn trắng, đã lọc sách trắng).

Khác cam3R3A_merge (bd01130):
  - ảnh thật: phngnam296/edgeai-cam-real2/khan_real = cam_real (1) 1075 + camtest_104_166 62 + iphone 85 = 1222 ảnh,
    đã XOÁ nhãn sách/vở/tài liệu trắng, sửa 8 nhãn book->box, bỏ 1 nhãn trên laptop. Có 38 ảnh nền không nhãn.
  - tách ~10% ảnh thật làm val (theo ĐOẠN 25 ảnh liền nhau, không bốc ngẫu nhiên) -> camval, KHÔNG train.
  - thầy khởi tạo từ thầy R3B, 25 epoch; trò khởi tạo từ trò R3B, 45 epoch, distill 2.0.
  - in mAP50 của thầy trên camval sau mỗi epoch.
Biến thể: REAL_REP (số lần lặp ảnh thật mỗi epoch) đặt ở dưới; cam4A = 3, cam4B = 6.
cam4C: như cam4B (x6) nhưng KHÔNG thầy / không distill, trò 70 epoch.
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

VARIANT = "cam4C"
USE_KD = False            # cam4C: trò tự học (không thầy), khởi tạo từ trò R3B
REAL_REP = 6
QUICK = False             # True: 2 epoch mỗi giai đoạn để thử pipeline
T_EPOCHS = 2 if QUICK else 25
S_EPOCHS = 2 if QUICK else 70
T_LR = 0.002
S_LR = 0.002
BATCH = 64
DIST_SCALE = (1.0, 1.0, 1.5)
KD_CLS = KD_BOX = 2.0
SPLIT_SEED = 42
VAL_BLOCK, VAL_EVERY, VAL_PICK = 25, 10, 5     # đoạn 25 ảnh; cứ 10 đoạn lấy đoạn thứ 5 làm val (~10%)
AUG = dict(hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.5, fliplr=0.5, flipud=0.5, close_mosaic=10)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199"])

ALL_FILES, ALL_DIRS = [], []
for root, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if "coco-2017" not in d.lower()]
    ALL_DIRS += [Path(root) / d for d in dirs]
    ALL_FILES += [Path(root) / f for f in files]
print(f"/kaggle/input: {len(ALL_FILES)} file", flush=True)


def find(name, must_contain=""):
    hits = sorted(p for p in ALL_FILES if p.name == name and must_contain in p.as_posix().lower())
    assert hits, f"khong thay {name} ({must_contain!r}) trong /kaggle/input"
    return hits[0]


def find_dir(name):
    hits = sorted(p for p in ALL_DIRS if p.name == name and (p / "images").is_dir())
    assert hits, f"khong thay thu muc {name}/images"
    return hits[0]


COMP = sorted(p.parent for p in ALL_FILES if p.name == "classes.txt" and (p.parent / "train" / "images").is_dir())[0]
TOOLS = find("yolo_reach.py", "edgeai-tools3").parent
sys.path.insert(0, str(TOOLS))
import yolo_reach  # noqa: E402

print("tam voi:", yolo_reach.apply(DIST_SCALE), "| box toi da:", yolo_reach.max_box(), flush=True)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402

W = Path("/kaggle/working")
TMP, DS = Path("/tmp/gen"), Path("/tmp/ds")
for d in (TMP, DS):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
CLASSES = [l.strip() for l in (COMP / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
names = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]"


def mkset(d, items):
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    for ip, lp in items:
        os.symlink(ip, d / "images" / ip.name)
        (d / "labels" / (ip.stem + ".txt")).write_text(lp.read_text() if lp.exists() else "")
    return d


t0 = time.time()
# ------------------------------------------------------------------ 1) BTC: val 20% + bản mô phỏng của 80% train
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
n_val = int(len(imgs) * 0.2)
val_btc, train_btc = imgs[:n_val], imgs[n_val:]
btc_val = mkset(TMP / "btc_val", [(p, tr_lbl / (p.stem + ".txt")) for p in val_btc])
tsim = find_dir("train_sim")
stems = {p.stem for p in train_btc}
btc_sim = mkset(TMP / "btc_sim", [(p, tsim / "labels" / (p.stem + ".txt")) for p in sorted((tsim / "images").iterdir())
                                  if p.stem.rsplit("_sim", 1)[0] in stems])

# ------------------------------------------------------------------ 2) gen5 (Roboflow + COCO/LVIS/OI/TACO dán vật)
GEN_MAN = sorted((p for p in ALL_FILES if re.fullmatch(r"gen\d+_manifest\.json", p.name)),
                 key=lambda p: int(re.findall(r"\d+", p.name)[0]), reverse=True)
assert GEN_MAN, "khong thay gen5_manifest.json"
print("dung", GEN_MAN[0], flush=True)
for part in ("rf_train", "rf_val", "coco"):
    with tarfile.open(GEN_MAN[0].parent / GEN_MAN[0].name.replace("manifest.json", f"{part}.tar")) as tar:
        tar.extractall(TMP)
RF_TRAIN = sorted((TMP / "rf_train").glob("p*"))
RF_VAL = sorted((TMP / "rf_val").glob("p*"))
COCO_OUT = sorted((TMP / "coco").glob("p*"))

# ------------------------------------------------------------------ 3) ảnh camera thật -> train / camval theo đoạn
real_root = None
for p in ALL_DIRS:
    if "edgeai-cam-real2" in p.as_posix().lower() and p.name == "khan_real" and (p / "images").is_dir():
        real_root = p
if real_root is None:
    zf = find("khan_real.zip", "edgeai-cam-real2")
    with zipfile.ZipFile(zf) as z:
        z.extractall(TMP / "real")
    real_root = next(p.parent for p in (TMP / "real").rglob("images") if p.is_dir())
print("anh that:", real_root, flush=True)


def order(p):   # số khung trước (0..1074), rồi camtest_*, rồi iph_*
    s = p.stem
    if s.isdigit():
        return (0, int(s))
    m = re.findall(r"\d+", s)
    return (1 if s.startswith("camtest") else 2, int(m[-1]) if m else 0)


real_imgs = sorted((p for p in (real_root / "images").iterdir() if p.is_file()), key=order)
real_tr, real_va = [], []
for i, p in enumerate(real_imgs):
    (real_va if (i // VAL_BLOCK) % VAL_EVERY == VAL_PICK else real_tr).append(p)
real_train = mkset(TMP / "real_train", [(p, real_root / "labels" / (p.stem + ".txt")) for p in real_tr])
real_val = mkset(TMP / "real_val", [(p, real_root / "labels" / (p.stem + ".txt")) for p in real_va])
(W / f"{VARIANT}_camval_list.txt").write_text("\n".join(p.name for p in real_va) + "\n")
print(f"anh that: {len(real_tr)} train, {len(real_va)} val", flush=True)


# ------------------------------------------------------------------ 4) gộp vào /tmp/ds
def link_all(src, split, prefix):
    (DS / "images" / split).mkdir(parents=True, exist_ok=True)
    (DS / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((src / "images").iterdir()):
        if not p.is_file():
            continue
        lp = src / "labels" / (p.stem + ".txt")
        os.symlink(p.resolve(), DS / "images" / split / f"{prefix}_{p.name}")
        (DS / "labels" / split / f"{prefix}_{p.stem}.txt").write_text(lp.read_text() if lp.exists() else "")
        n += 1
    return n


cnt = {
    "btc_sim": link_all(btc_sim, "train", "btcsim"),
    "rf": sum(link_all(d, "train", f"rf{k}") for k, d in enumerate(RF_TRAIN)),
    "coco": sum(link_all(d, "train", f"coco{k}") for k, d in enumerate(COCO_OUT)),
    "real_x%d" % REAL_REP: sum(link_all(real_train, "train", f"real{r}") for r in range(REAL_REP)),
    "val_btc": link_all(btc_val, "val", "btc"),
    "val_rf": sum(link_all(d, "rfval", f"rf{k}") for k, d in enumerate(RF_VAL)),
    "val_cam": link_all(real_val, "camval", "cv"),
}
real55 = find("split.json").parent
cnt["test_real55"] = link_all(real55 / "train", "real", "a") + link_all(real55 / "test", "real", "b")
cnt["test_btcmau"] = link_all(find_dir("btc_samples"), "btcmau", "m")
n_train = cnt["btc_sim"] + cnt["rf"] + cnt["coco"] + cnt["real_x%d" % REAL_REP]
print("Dataset:", cnt, "| ti le anh that %.1f%%" % (100 * cnt["real_x%d" % REAL_REP] / n_train),
      "| unpack %.1f phut" % ((time.time() - t0) / 60), flush=True)


def yaml_for(val):
    v = "[" + ", ".join(val) + "]" if isinstance(val, list) else val
    return f"path: {DS}\ntrain: images/train\nval: {v}\nnames: {names}\n"


(DS / "data.yaml").write_text(yaml_for(["images/val", "images/rfval", "images/camval"]))
SPLITS = {"camval": "images/camval", "btc": "images/val", "rf": "images/rfval", "real": "images/real",
          "btcmau": "images/btcmau"}
for k, v in SPLITS.items():
    (DS / f"{k}.yaml").write_text(yaml_for(v))


def val_one(weights, split, tag):
    r = YOLO(str(weights)).val(data=str(DS / f"{split}.yaml"), imgsz=320, batch=64, device=0, plots=False,
                               verbose=False, project="/tmp/val", name=f"{tag}_{split}", exist_ok=True)
    return {"map50": round(float(r.box.map50), 4),
            "per_class": {CLASSES[int(c)]: round(float(a), 3) for c, a in zip(r.box.ap_class_index, r.box.ap50)}}


def evaluate(weights, tag):
    out = {}
    for split in SPLITS:
        out[split] = val_one(weights, split, tag)
        print(f"[{tag}] {split:6} mAP50={out[split]['map50']} {out[split]['per_class']}", flush=True)
    return out


def load_matching(model, sd):
    own = model.state_dict()
    take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    return len(take), len(own)


RES = {"variant": VARIANT, "counts": cnt, "real_rep": REAL_REP, "t_epochs": T_EPOCHS, "s_epochs": S_EPOCHS,
       "kd": KD_CLS if USE_KD else 0, "init": "R3B", "dist_scale": list(DIST_SCALE), "teacher_camval_by_epoch": []}

TEACHER_W = None   # cam4C: không thầy


# ------------------------------------------------------------------ 6) TRÒ P2P4: từ trò R3B, distill
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
    print(f"Distill tu {TEACHER_W.name} ({sum(p.numel() for p in t.parameters()):,} tham so), he so {KD_CLS}",
          flush=True)


t0 = time.time()
student = YOLO(str(find("student_p2p4.yaml", "edgeai-tools3")))
student_init = find("yolo_cam3R3B_last.pt", "edgeai-cam-real2")
print("Student: nap %d/%d tensor tu tro R3B" % load_matching(student.model, YOLO(str(student_init)).model.state_dict()),
      flush=True)
student.load(student.model)
n_params = sum(p.numel() for p in student.model.parameters())
print(f"Student P2P4: {n_params:,} tham so", flush=True)
assert n_params <= 200_000
if USE_KD:
    student.add_callback("on_train_start", attach_kd)
student.train(data=str(DS / "data.yaml"), epochs=S_EPOCHS, imgsz=320, batch=BATCH, workers=4, device=0,
              pretrained=True, optimizer="SGD", lr0=S_LR, lrf=0.05, cos_lr=True, warmup_epochs=1,
              warmup_bias_lr=S_LR, project=str(W), name="student", exist_ok=True, plots=False, patience=0, seed=0,
              **AUG)
print(f"tro xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)

for wn in ("best.pt", "last.pt"):
    RES["student_" + wn] = evaluate(W / "student" / "weights" / wn, "student_" + wn)
RES["r3b_student"] = evaluate(student_init, "r3b")

(W / "results_cam.json").write_text(json.dumps(RES, indent=2))
for src, dst in (("student/weights/best.pt", f"yolo_{VARIANT}_best.pt"),
                 ("student/weights/last.pt", f"yolo_{VARIANT}_last.pt"),
                 ):
    shutil.copy(W / src, W / dst)
shutil.rmtree(W / "teacher", ignore_errors=True)
shutil.rmtree(W / "student", ignore_errors=True)
print(f"XONG {VARIANT}", json.dumps({k: {s: v[s]["map50"] for s in v} for k, v in RES.items()
                                       if isinstance(v, dict) and "camval" in v}), flush=True)
