"""THỬ NGHIỆM (KẾT QUẢ ÂM, không đưa vào firmware): chấm lại CHUỘT <-> ĐIỆN THOẠI theo KÍCH THƯỚC THẬT.
Trên ảnh khăn (2026-11-27): chuột hiếm khi nhầm sang điện thoại (11/193); lỗi chính là SÓT chuột đen trên nền trắng
và box chuột bao cả dây (nhãn chỉ thân) -> chấm lại/lọc theo kích thước không cải thiện (test 20/31 -> 16-17/31).

Mỗi ảnh: ước lượng px/cm = trung vị (cạnh dài box / cạnh dài thật) của các vật CHẮC CHẮN (điểm >= --anchor-conf,
không chạm mép) thuộc phone/mouse/toy_car/book/box (bỏ chai, cốc vì phụ thuộc cách đặt), KHÔNG tính chính box đang xét.
Box phone/mouse: đổi px -> cm, so khoảng cách log tới 16 cm (điện thoại) và 10 cm (chuột); khớp lớp kia rõ hơn
(--margin) và model cho lớp kia >= --pmin điểm -> đổi lớp.
Hậu xử lý giống firmware: ngưỡng 0.45 (chai 0.30), NMS 0.30 trong từng lớp; giữ ĐỦ điểm 7 lớp mỗi box.

    python tools/size_rescore.py --weights kaggle_runs/cam3A/out/yolo_cam3A_last.pt
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import yolo_reach  # noqa: E402

NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
LONG_CM = {"phone": 16, "mouse": 10, "toy_car": 7, "book": 20, "box": 13}   # đồ thi (đo 2026-11-27)
THR = np.array([0.30, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45], np.float32)
PH, MO = NAMES.index("phone"), NAMES.index("mouse")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def detect(net, path):
    """-> list (xyxy chuẩn hoá, vector điểm 7 lớp, lớp) sau ngưỡng + NMS trong từng lớp"""
    from torchvision.ops import batched_nms
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (320, 240))
    x = torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        y = net(x)
    y = (y[0] if isinstance(y, (list, tuple)) else y)[0]            # (4+nc, N): cx, cy, w, h (px), điểm lớp
    sc = y[4:].T.numpy()
    c = sc.argmax(1)
    keep = sc[np.arange(len(c)), c] >= THR[c]
    if not keep.any():
        return []
    b = y[:4].T.numpy()[keep]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    s, cc = sc[keep], c[keep]
    k = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(s[np.arange(len(cc)), cc]), torch.from_numpy(cc), 0.30).numpy()
    return [(xyxy[i] / [320, 240, 320, 240], s[i], int(cc[i])) for i in k]


def long_px(b):
    return max((b[2] - b[0]) * 320, (b[3] - b[1]) * 240)


def edge(b):
    return b[0] < 0.01 or b[1] < 0.01 or b[2] > 0.99 or b[3] > 0.99


def rescore(dets, margin, pmin, anchor_conf):
    out = []
    for i, (b, s, c) in enumerate(dets):
        if c in (PH, MO):
            samp = [long_px(bo) / LONG_CM[NAMES[co]] for j, (bo, so, co) in enumerate(dets)
                    if j != i and NAMES[co] in LONG_CM and so[co] >= anchor_conf and not edge(bo)]
            if samp:
                L = long_px(b) / float(np.median(samp))
                dp, dm = abs(math.log(L / 16)), abs(math.log(L / 10))
                if c == PH and dm + margin < dp and s[MO] >= pmin:
                    c = MO
                elif c == MO and dp + margin < dm and s[PH] >= pmin:
                    c = PH
        out.append((b, s, c))
    return out


def load_gt(p):
    g = []
    if os.path.exists(p):
        for l in open(p):
            q = l.split()
            if len(q) == 5:
                c, x, y, w, h = map(float, q)
                g.append((np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]), int(c)))
    return g


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def score(all_dets, gts):
    """chuột/điện thoại thật: đúng / nhầm sang lớp kia / sót; + box mouse/phone dư (không khớp vật cùng lớp)"""
    r = Counter()
    for dets, gt in zip(all_dets, gts):
        used = set()
        for gb, gc in gt:
            if gc not in (PH, MO):
                continue
            m = max(((iou(gb, b), j) for j, (b, s, c) in enumerate(dets) if j not in used), default=(0, -1))
            if m[0] < 0.5:
                r[f"{NAMES[gc]}_sot"] += 1
                continue
            used.add(m[1])
            pc = dets[m[1]][2]
            r[f"{NAMES[gc]}_{'dung' if pc == gc else ('thanh_' + NAMES[pc])}"] += 1
        for j, (b, s, c) in enumerate(dets):
            if c in (PH, MO) and not any(gc == c and iou(gb, b) >= 0.5 for gb, gc in gt):
                r[f"du_{NAMES[c]}"] += 1
    return r


def summary(r):
    md, pd_ = r["mouse_dung"], r["phone_dung"]
    mt = md + r["mouse_thanh_phone"] + r["mouse_sot"] + sum(v for k, v in r.items() if k.startswith("mouse_thanh_") and k != "mouse_thanh_phone")
    pt = pd_ + r["phone_thanh_mouse"] + r["phone_sot"] + sum(v for k, v in r.items() if k.startswith("phone_thanh_") and k != "phone_thanh_mouse")
    return (f"chuột đúng {md}/{mt} (thành phone {r['mouse_thanh_phone']}) | điện thoại đúng {pd_}/{pt} "
            f"(thành mouse {r['phone_thanh_mouse']}) | box dư: mouse {r['du_mouse']} phone {r['du_phone']}"), md + pd_ - r["du_mouse"] - r["du_phone"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="kaggle_runs/cam3A/out/yolo_cam3A_last.pt")
    ap.add_argument("--dist-scale", default="1,1,1.5")
    ap.add_argument("--val", default="data/cam_new_train")
    ap.add_argument("--test", default="data/cam_new_test")
    a = ap.parse_args()
    yolo_reach.apply(tuple(map(float, a.dist_scale.split(","))))
    from ultralytics import YOLO
    net = YOLO(a.weights).model.float().eval()
    sets = {}
    for name, d in (("val", a.val), ("test", a.test)):
        ims = sorted(glob.glob(f"{d}/images/*.jpg"))
        sets[name] = ([detect(net, p) for p in ims], [load_gt(f"{d}/labels/{Path(p).stem}.txt") for p in ims])
        print(f"{name}: {len(ims)} ảnh", flush=True)
    for name in ("val", "test"):
        print(f"\n[{name}] GỐC          : {summary(score(*sets[name]))[0]}")
    best = None
    for margin, pmin, ac in itertools.product((0.0, 0.1, 0.2, 0.3), (0.0, 0.02, 0.05, 0.1), (0.5, 0.6, 0.7)):
        sc = summary(score([rescore(d, margin, pmin, ac) for d in sets["val"][0]], sets["val"][1]))[1]
        if best is None or sc > best[0]:
            best = (sc, margin, pmin, ac)
    _, margin, pmin, ac = best
    print(f"\nchọn trên VAL: margin {margin}, pmin {pmin}, anchor_conf {ac}")
    for name in ("val", "test"):
        r = score([rescore(d, margin, pmin, ac) for d in sets[name][0]], sets[name][1])
        print(f"[{name}] CHẤM LẠI    : {summary(r)[0]}")


if __name__ == "__main__":
    main()
