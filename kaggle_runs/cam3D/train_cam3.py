# cam3: TONG QUAT HOA cho chung ket (do vat/canh hoan toan moi). KHONG train tren anh camera that (chi de test).
# Du lieu train (tao ngay tren Kaggle):
#   - BTC train 80%: CHI ban mo phong OV7670 (giam anh 3D, bot hoc thuoc)
#   - Roboflow: dien nhan thieu bang YOLO11x COCO -> zoom_out (da sua: vat phai DU HINH, khong bi cat) -> mo phong
#   - COCO 2017 + LVIS v1: cat vat theo MASK (bottle/cup/phone/book/mouse + box, coc thuy tinh, mug...), dan len nen BTC dung ti le
#     camera, xoay moi huong -> hang nghin kieu vat khac nhau -> mo phong
# Test (khong bao gio train): 9 anh mau BTC that, 55 anh camera that (data/cam_real da kiem nhan).
# Val chon best: BTC val + Roboflow val (5% anh goc giu rieng TRUOC khi ghep mosaic -> khong ro ri).
import os, sys, subprocess, random, shutil, time, json
from pathlib import Path

QUICK = os.environ.get("QUICK", "0") == "1"
# Thu nghiem song song: moi ban chi doi 1 yeu to so voi A (de biet yeu to nao co tac dung).
#  A: goc. Thay + tro deu khoi tao tu cam1.
#  B: thay LON (backbone YOLO11m pretrained COCO, ~10M) -> kien thuc rong hon cho distill.
#  C: tro train TU DAU (khong nap cam1) + nhieu epoch -> xem cam1 co keo theo "hoc thuoc" anh 3D khong.
#  D: tro P3/P4 (bo P2, 161k, -38% tinh toan); distill tu thay P2P4 (chi lay 2 muc P3/P4 cua thay).
VARIANT = "D"
V = {"A": dict(teacher="cam1", t_epochs=30, t_lr=0.005, s_yaml="p2p4", s_init="cam1", s_epochs=60, s_lr=0.002),
     "B": dict(teacher="yolo11m", t_epochs=40, t_lr=0.01, s_yaml="p2p4", s_init="cam1", s_epochs=60, s_lr=0.002),
     "C": dict(teacher="cam1", t_epochs=30, t_lr=0.005, s_yaml="p2p4", s_init=None, s_epochs=100, s_lr=0.01),
     "D": dict(teacher="cam1", t_epochs=30, t_lr=0.005, s_yaml="p3p4", s_init="cam1", s_epochs=60, s_lr=0.004),
     }[VARIANT]
T_EPOCHS = 2 if QUICK else V["t_epochs"]
S_EPOCHS = 2 if QUICK else V["s_epochs"]
N_COCO = 150 if QUICK else 4000
RF_RATIO = 0.03 if QUICK else 0.6
BATCH = 64
DIST_SCALE = (1.0, 1.0, 1.5)
KD_CLS, KD_BOX = 1.0, 1.0
SPLIT_SEED = 42
AUG = dict(hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.5, fliplr=0.5, flipud=0.5, close_mosaic=10)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.199", "pycocotools"])

INP = Path("/kaggle/input")
# Kaggle moi: /kaggle/input/datasets/<chu>/<ten>/..., /kaggle/input/competitions/<ten>/..., notebook output...
# -> duyet 1 lan toan bo /kaggle/input, BO QUA thu muc COCO (160k file), luu danh sach file/thu muc.
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
print(f"/kaggle/input: {len(ALL_FILES)} file (khong tinh COCO), COCO: {COCO_DIRS}", flush=True)


def find(name):
    hits = sorted(p for p in ALL_FILES if p.name == name)
    assert hits, f"khong thay {name} trong /kaggle/input"
    return hits[0]


def find_dir(name):
    hits = sorted(p for p in ALL_DIRS if p.name == name and (p / "images").is_dir())
    assert hits, f"khong thay thu muc {name}/images"
    return hits[0]


def find_coco():
    for c in COCO_DIRS:
        for ann in list(c.glob("annotations/instances_train2017.json")) + list(c.glob("*/annotations/instances_train2017.json")):
            img = ann.parent.parent / "train2017"
            if img.is_dir():
                return ann, img
    raise AssertionError(f"khong thay COCO 2017 trong {COCO_DIRS}")


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
rf_items = []
for rname in ("rf_merged", "rf_hotwheels"):   # rf_hotwheels: xe do choi Hot Wheels (neu co trong dataset)
    try:
        rfm = find_dir(rname)
    except AssertionError:
        print(f"(khong co {rname})", flush=True)
        continue
    its = [(p, rfm / "labels" / (p.stem + ".txt")) for p in sorted((rfm / "images").iterdir())]
    print(f"{rname}: {len(its)} anh", flush=True)
    rf_items += its
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
croot = cimg.parent                     # chua train2017/ + val2017/ (anh LVIS nam o ca hai)
assert (croot / "val2017").is_dir(), f"thieu {croot}/val2017"
lvis = []
for part in ("train", "val"):           # LVIS v1: cung anh COCO, them box/carton/coc thuy tinh/mug... co mask
    z = TMP / f"lvis_v1_{part}.json.zip"
    subprocess.check_call(["wget", "-q", "-O", str(z), f"https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{part}.json.zip"])
    subprocess.check_call(["unzip", "-q", "-o", str(z), "-d", str(TMP)])
    z.unlink()
    lvis.append(TMP / f"lvis_v1_{part}.json")
run(TOOLS / "coco_paste.py", "--ann", ann, *[x for p in lvis for x in ("--ann", p)], "--images", croot,
    "--bg", btc_train, "--out", TMP / "coco_paste", "--n", N_COCO, "--preview", W / "coco_paste_preview.jpg")
for p in lvis:
    p.unlink()
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


# anh 3D BTC: chi ban mo phong camera (bo ban goc) -> anh 3D tu ~46% xuong ~30% so vat, bot hoc thuoc vai mo hinh 3D
cnt = {"btc_sim": link_all(btc_sim, "train", "btcsim"),
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


RES = {"variant": VARIANT, "cfg": V}
TEACHER_YAML_S = (TOOLS / "teacher_p2p4s.yaml").read_text().replace(chr(13), "")   # file co the la CRLF
STUDENT_P3P4 = """nc: 7
reg_max: 4
scales:
  s: [0.33, 0.25, 256]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SCDown, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, SCDown, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, SPPF, [512, 5]]
  - [-1, 1, nn.Identity, []]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, SCDown, [256, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]
  - [[11, 14], 1, Detect, [nc]]
"""


def load_matching(model, sd, remap=None):
    """nap trong so theo ten (+ doi ten lop qua remap), chi lay tensor cung shape"""
    own = model.state_dict()
    take = {}
    for k, v in sd.items():
        k2 = k
        for a, b in (remap or {}).items():
            if k.startswith(a):
                k2 = b + k[len(a):]
                break
        if k2 in own and own[k2].shape == v.shape:
            take[k2] = v
    model.load_state_dict(take, strict=False)
    return len(take), len(own)


# ------------------------------------------------------------------ 1) THAY
t0 = time.time()
if V["teacher"] == "cam1":
    tcfg = Path("/tmp/teacher_p2p4s.yaml")
    tcfg.write_text(TEACHER_YAML_S)
    teacher = YOLO(str(tcfg)).load(str(find("yolo_cam_teacher.pt")))
else:   # yolo11m: cung bo cuc P2P4/reg_max 4 (de distill), scale m, backbone 0-6 tu YOLO11m COCO
    tcfg = Path("/tmp/teacher_p2p4m.yaml")   # ten co 'm' -> Ultralytics chon scale m
    tcfg.write_text(TEACHER_YAML_S.replace("scales:\n  s: [0.5, 0.5, 1024]", "scales:\n  m: [0.5, 1.0, 512]"))
    assert "m: [0.5, 1.0, 512]" in tcfg.read_text(), "khong doi duoc scale thay"
    teacher = YOLO(str(tcfg))
    coco_sd = YOLO("yolo11m.pt").model.state_dict()
    bb = {k: v for k, v in coco_sd.items() if int(k.split(".")[1]) <= 6}
    print("thay yolo11m: nap %d/%d tensor backbone COCO" % load_matching(teacher.model, bb), flush=True)
    teacher.load(teacher.model)
print(f"thay: {sum(p.numel() for p in teacher.model.parameters()):,} tham so", flush=True)
teacher.train(data=str(DS / "data.yaml"), epochs=T_EPOCHS, imgsz=320, batch=BATCH, workers=4, device=0,
              pretrained=True, optimizer="SGD", lr0=V["t_lr"], lrf=0.05, cos_lr=True,
              warmup_epochs=1 if V["teacher"] == "cam1" else 3,
              warmup_bias_lr=V["t_lr"] if V["teacher"] == "cam1" else 0.1, project=str(W), name="teacher",
              exist_ok=True, plots=False, patience=0, seed=0, **AUG)
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
        n_s = preds["scores"].shape[-1]
        if tp["scores"].shape[-1] != n_s:   # ban D: tro P3/P4, thay P2/P3/P4 -> chi lay cac o P3/P4 (cuoi danh sach)
            tp = {"scores": tp["scores"][..., -n_s:], "boxes": tp["boxes"][..., -n_s:]}
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
    tb, sb = t_out[1]["boxes"].shape, s_out[1]["boxes"].shape
    assert tb[:2] == sb[:2] and tb[2] >= sb[2], f"Thay {tuple(tb)} va tro {tuple(sb)} khac bo cuc dau ra"
    trainer.model.criterion = KDLoss(trainer.model, t)
    print(f"distill tu {TEACHER_W.name} ({sum(p.numel() for p in t.parameters()):,} tham so)", flush=True)


t0 = time.time()
S_SCALE = (1.0, 1.0, 1.5) if V["s_yaml"] == "p2p4" else (1.0, 1.5)   # tam voi x1.5 luon o muc cuoi (P4)
yolo_reach.apply(S_SCALE)
if V["s_yaml"] == "p2p4":
    student = YOLO(str(find("student_p2p4.yaml")))
else:
    scfg = Path("/tmp/student_p3p4.yaml")
    scfg.write_text(STUDENT_P3P4)
    student = YOLO(str(scfg))
if V["s_init"] == "cam1":
    cam1_sd = YOLO(str(find("yolo_cam_best.pt"))).model.state_dict()
    # P3P4: lop 12/14 = lop 18/20 cua P2P4; Detect muc 0/1 = muc 1/2 cua P2P4
    remap = {} if V["s_yaml"] == "p2p4" else {
        "model.18.": "model.12.", "model.20.": "model.14.", "model.21.cv2.1.": "model.15.cv2.0.",
        "model.21.cv2.2.": "model.15.cv2.1.", "model.21.cv3.1.": "model.15.cv3.0.", "model.21.cv3.2.": "model.15.cv3.1.",
        "model.21.dfl.": "model.15.dfl."}
    if V["s_yaml"] != "p2p4":   # bo cac lop P2P4 khong co ten tuong ung (12..17, 21 muc 0) truoc khi doi ten
        cam1_sd = {k: v for k, v in cam1_sd.items()
                   if int(k.split(".")[1]) <= 11 or any(k.startswith(a) for a in remap)}
    print("tro: nap %d/%d tensor tu cam1" % load_matching(student.model, cam1_sd, remap), flush=True)
    student.load(student.model)
n = sum(p.numel() for p in student.model.parameters())
print(f"tro {V['s_yaml']}: {n:,} tham so, khoi tao: {V['s_init'] or 'tu dau'}", flush=True)
assert n <= 200_000
student.add_callback("on_train_start", attach_kd)
student.train(data=str(DS / "data.yaml"), epochs=S_EPOCHS, imgsz=320, batch=BATCH, workers=4, device=0,
              pretrained=True, optimizer="SGD", lr0=V["s_lr"], lrf=0.05, cos_lr=True,
              warmup_epochs=1 if V["s_init"] else 3, warmup_bias_lr=V["s_lr"] if V["s_init"] else 0.1,
              project=str(W), name="student", exist_ok=True, plots=False, patience=0, seed=0, **AUG)
print(f"tro xong sau {(time.time() - t0) / 60:.1f} phut", flush=True)
for wn in ("best.pt", "last.pt"):
    RES["student_" + wn] = evaluate(W / "student" / "weights" / wn, "tro_" + wn)
yolo_reach.apply((1.0, 1.0, 1.5))       # model cu deu P2P4
RES["cam1_student"] = evaluate(find("yolo_cam_best.pt"), "cam1")
RES["cam2_student"] = evaluate(find("yolo_cam2_last.pt"), "cam2(da hoc anh that)")

(W / "results_cam.json").write_text(json.dumps(RES, indent=1))
for src, dst in (("student/weights/best.pt", f"yolo_cam3{VARIANT}_best.pt"),
                 ("student/weights/last.pt", f"yolo_cam3{VARIANT}_last.pt"),
                 ("teacher/weights/last.pt", f"yolo_cam3{VARIANT}_teacher.pt")):
    shutil.copy(W / src, W / dst)
shutil.rmtree(W / "teacher", ignore_errors=True)
shutil.rmtree(W / "student", ignore_errors=True)
print("XONG", json.dumps({k: {s: v[s]["map50"] for s in v} for k, v in RES.items()}), flush=True)
