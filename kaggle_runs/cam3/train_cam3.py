# cam3: TONG QUAT HOA cho chung ket (do vat/canh hoan toan moi). KHONG train tren anh camera that (chi de test).
# Du lieu train (tao ngay tren Kaggle):
#   - BTC train 80% + ban mo phong OV7670
#   - Roboflow: dien nhan thieu bang YOLO11x COCO -> zoom_out (da sua: vat phai DU HINH, khong bi cat) -> mo phong
#   - COCO 2017: cat vat theo MASK (bottle/cup/wine glass/cell phone/book/mouse), dan len nen BTC dung ti le
#     camera, xoay moi huong -> hang nghin kieu vat khac nhau -> mo phong
# Test (khong bao gio train): 9 anh mau BTC that, 55 anh camera that (data/cam_real da kiem nhan).
# Val chon best: BTC val + Roboflow val (5% anh goc giu rieng TRUOC khi ghep mosaic -> khong ro ri).
import os, sys, subprocess, random, shutil, time, json
from pathlib import Path

QUICK = os.environ.get("QUICK", "0") == "1"
T_EPOCHS = 2 if QUICK else 30
S_EPOCHS = 2 if QUICK else 60
N_COCO = 150 if QUICK else 4000
RF_RATIO = 0.03 if QUICK else 0.6
BATCH = 64
DIST_SCALE = (1.0, 1.0, 1.5)
KD_CLS, KD_BOX = 1.0, 1.0
SPLIT_SEED = 42
AUG = dict(hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.5, fliplr=0.5, flipud=0.5, close_mosaic=10)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199", "pycocotools"])

INP = Path("/kaggle/input")
SMALL = [d for d in INP.iterdir() if "coco-2017" not in d.name]      # khong duyet 160k file COCO moi lan tim


def find(name):
    hits = sorted(h for d in SMALL for h in d.rglob(name))
    assert hits, f"khong thay {name} trong /kaggle/input"
    return hits[0]


def find_dir(name):
    hits = sorted(p for d in SMALL for p in d.rglob(name) if p.is_dir() and (p / "images").is_dir())
    assert hits, f"khong thay thu muc {name}/images"
    return hits[0]


def find_coco():
    for ann in INP.glob("*coco-2017*/**/annotations/instances_train2017.json"):
        img = ann.parent.parent / "train2017"
        if img.is_dir():
            return ann, img
    raise AssertionError("khong thay COCO 2017 (instances_train2017.json + train2017/)")


TOOLS = find("coco_paste.py").parent
COMP = find("classes.txt").parent
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
CLASSES = [l.strip() for l in (COMP / "classes.txt").read_text().splitlines() if l.strip()]
names = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]"
PROFILE = TOOLS / "ov7670_profile.json"
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(DS, ignore_errors=True)


def run(*args):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.check_call([sys.executable, *map(str, args)])


def mkset(d, items):
    """items: [(anh, nhan)] -> d/images, d/labels (symlink anh, chep nhan)"""
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
# ------------------------------------------------------------------ BTC
tr_img, tr_lbl = COMP / "train" / "images", COMP / "train" / "labels"
imgs = sorted(p for p in tr_img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
random.Random(SPLIT_SEED).shuffle(imgs)
n_val = max(1, int(len(imgs) * 0.2))
val_btc, train_btc = imgs[:n_val], imgs[n_val:]
btc_train = mkset(TMP / "btc_train", [(p, tr_lbl / (p.stem + ".txt")) for p in train_btc])
btc_val = mkset(TMP / "btc_val", [(p, tr_lbl / (p.stem + ".txt")) for p in val_btc])
tsim = find_dir("train_sim")
stems = {p.stem for p in train_btc}
btc_sim = mkset(TMP / "btc_sim", [(p, tsim / "labels" / (p.stem + ".txt")) for p in sorted((tsim / "images").iterdir())
                                  if p.stem.rsplit("_sim", 1)[0] in stems])

# ------------------------------------------------------------------ Roboflow: dien nhan thieu -> zoom (vat du hinh) -> sim
rfm = find_dir("rf_merged")
rf_items = [(p, rfm / "labels" / (p.stem + ".txt")) for p in sorted((rfm / "images").iterdir())]
random.Random(1).shuffle(rf_items)
if QUICK:
    rf_items = rf_items[:600]
n_rv = max(1, len(rf_items) // 20)
rf_parts = {"rfv": rf_items[:n_rv], "rft": rf_items[n_rv:]}
COCO_IDS = {39: 0, 40: 1, 41: 1, 67: 2, 73: 3, 64: 4}   # bottle, wine glass->cup, cup, cell phone, book, mouse
filler = YOLO("yolo11x.pt")


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


n_fill = 0
for part, items in rf_parts.items():
    d = mkset(TMP / f"{part}_filled", items)
    ims = sorted((d / "images").iterdir())
    for k in range(0, len(ims), 64):
        for r in filler.predict([str(p) for p in ims[k:k + 64]], imgsz=640, conf=0.5, classes=list(COCO_IDS),
                                verbose=False, device=0):
            lp = d / "labels" / (Path(r.path).stem + ".txt")
            old = [list(map(float, l.split())) for l in lp.read_text().split("\n") if l.strip()]
            ob = [(o[1] - o[3] / 2, o[2] - o[4] / 2, o[1] + o[3] / 2, o[2] + o[4] / 2) for o in old]
            add = []
            for c, b in zip(r.boxes.cls.int().tolist(), r.boxes.xyxyn.tolist()):
                if all(iou(b, o) < 0.3 for o in ob):
                    add.append(f"{COCO_IDS[c]} {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {b[2] - b[0]:.6f} {b[3] - b[1]:.6f}\n")
                    ob.append(tuple(b))
            if add:
                with open(lp, "a") as f:
                    f.write(("" if not old or lp.read_text().endswith("\n") else "\n") + "".join(add))
                n_fill += len(add)
print(f"Roboflow: dien them {n_fill} box thieu nhan (YOLO11x COCO, conf 0.5)", flush=True)
del filler
torch.cuda.empty_cache()

classes_file = TMP / "classes.txt"
classes_file.write_text("\n".join(CLASSES) + "\n")
for part, ratio in (("rft", RF_RATIO), ("rfv", 1.0)):
    run(TOOLS / "zoom_out.py", "--src", TMP / f"{part}_filled", "--out", TMP / f"{part}_zoom", "--ratio", ratio,
        "--classes", classes_file, "--seed", 0 if part == "rft" else 1)
run(TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / "rft_zoom", "--out", TMP / "rft_sim",
    "--copies", 1, "--keep-orig")
run(TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / "rfv_zoom", "--out", TMP / "rfv_sim",
    "--copies", 1)

# ------------------------------------------------------------------ COCO: cat theo mask -> dan len nen BTC -> sim
ann, cimg = find_coco()
run(TOOLS / "coco_paste.py", "--ann", ann, "--images", cimg, "--bg", btc_train, "--out", TMP / "coco_paste",
    "--n", N_COCO, "--preview", W / "coco_paste_preview.jpg")
run(TOOLS / "sim_ov7670.py", "apply", "--profile", PROFILE, "--src", TMP / "coco_paste", "--out", TMP / "coco_sim",
    "--copies", 1, "--keep-orig")
print(f"tao du lieu xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)

# ------------------------------------------------------------------ gom vao /tmp/ds
def link_all(src, split, prefix):
    n = 0
    (DS / "images" / split).mkdir(parents=True, exist_ok=True)
    (DS / "labels" / split).mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "images").iterdir()):
        lp = src / "labels" / (p.stem + ".txt")
        os.symlink(p.resolve(), DS / "images" / split / f"{prefix}_{p.name}")
        dst = DS / "labels" / split / f"{prefix}_{p.stem}.txt"
        if lp.exists():
            os.symlink(lp.resolve(), dst)
        else:
            dst.write_text("")
        n += 1
    return n


cnt = {"btc": link_all(btc_train, "train", "btc"), "btc_sim": link_all(btc_sim, "train", "btcsim"),
       "rf": link_all(TMP / "rft_sim", "train", "rf"), "coco": link_all(TMP / "coco_sim", "train", "coco"),
       "val_btc": link_all(btc_val, "val", "btc"), "val_rf": link_all(TMP / "rfv_sim", "rfval", "rf")}
real = find("split.json").parent
cnt["test_real"] = link_all(real / "train", "real", "a") + link_all(real / "test", "real", "b")
cnt["test_btcmau"] = link_all(find_dir("btc_samples"), "btcmau", "m")
print("so anh:", cnt, flush=True)


def yaml_for(val):
    v = "[" + ", ".join(val) + "]" if isinstance(val, list) else val
    return f"path: {DS}\ntrain: images/train\nval: {v}\nnames: {names}\n"


(DS / "data.yaml").write_text(yaml_for(["images/val", "images/rfval"]))
for k, v in (("btc", "images/val"), ("rf", "images/rfval"), ("real", "images/real"), ("btcmau", "images/btcmau")):
    (DS / f"{k}.yaml").write_text(yaml_for(v))


def evaluate(weights, tag):
    out = {}
    for split in ("btc", "rf", "real", "btcmau"):
        r = YOLO(str(weights)).val(data=str(DS / f"{split}.yaml"), imgsz=320, batch=64, device=0, plots=False,
                                   verbose=False, project="/tmp/val", name=f"{tag}_{split}", exist_ok=True)
        out[split] = {"map50": round(float(r.box.map50), 4),
                      "per_class": {CLASSES[int(c)]: round(float(a), 3) for c, a in zip(r.box.ap_class_index, r.box.ap50)}}
        print(f"[{tag}] {split} mAP50 {out[split]['map50']} {out[split]['per_class']}", flush=True)
    return out


RES = {}
TEACHER_YAML = (TOOLS / "teacher_p2p4s.yaml")
# ------------------------------------------------------------------ 1) THAY (tu thay cam1)
t0 = time.time()
teacher = YOLO(str(TEACHER_YAML)).load(str(find("yolo_cam_teacher.pt")))
teacher.train(data=str(DS / "data.yaml"), epochs=T_EPOCHS, imgsz=320, batch=BATCH, workers=4, device=0,
              pretrained=True, optimizer="SGD", lr0=0.005, lrf=0.05, cos_lr=True, warmup_epochs=1,
              warmup_bias_lr=0.005, project=str(W), name="teacher", exist_ok=True, plots=False, patience=0, seed=0,
              **AUG)
TEACHER_W = W / "teacher" / "weights" / "last.pt"
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
assert n <= 200_000
student.add_callback("on_train_start", attach_kd)
student.train(data=str(DS / "data.yaml"), epochs=S_EPOCHS, imgsz=320, batch=BATCH, workers=4, device=0,
              pretrained=True, optimizer="SGD", lr0=0.002, lrf=0.05, cos_lr=True, warmup_epochs=1,
              warmup_bias_lr=0.002, project=str(W), name="student", exist_ok=True, plots=False, patience=0, seed=0,
              **AUG)
print(f"tro xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)
for wn in ("best.pt", "last.pt"):
    RES["student_" + wn] = evaluate(W / "student" / "weights" / wn, "tro_" + wn)
RES["cam1_student"] = evaluate(find("yolo_cam_best.pt"), "cam1")
RES["cam2_student"] = evaluate(find("yolo_cam2_last.pt"), "cam2(da hoc anh that)")

(W / "results_cam.json").write_text(json.dumps(RES, indent=1))
for src, dst in (("student/weights/best.pt", "yolo_cam3_best.pt"), ("student/weights/last.pt", "yolo_cam3_last.pt"),
                 ("teacher/weights/last.pt", "yolo_cam3_teacher.pt")):
    shutil.copy(W / src, W / dst)
shutil.rmtree(W / "teacher", ignore_errors=True)
shutil.rmtree(W / "student", ignore_errors=True)
print("XONG", json.dumps({k: {s: v[s]["map50"] for s in v} for k, v in RES.items()}), flush=True)
