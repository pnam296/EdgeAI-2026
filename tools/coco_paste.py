"""Cắt vật COCO theo MASK (đúng đường viền) rồi dán lên nền bàn (ảnh BTC) đúng tỉ lệ camera -> ảnh 320x240 + nhãn.

Mục tiêu: model thấy HÀNG NGHÌN kiểu chai/cốc/điện thoại/sách/chuột khác nhau, vật luôn đủ hình dạng (không bị cắt),
ở mọi hướng xoay (camera nhìn từ trên), thay vì học thuộc vài món đồ.

    python tools/coco_paste.py --ann .../instances_train2017.json --images .../train2017 \
        --bg data/train --out data/coco_paste --n 5000 --preview outputs/coco_paste.jpg

Chọn vật COCO: không crowd, cạnh bbox >= --min-px, không chạm mép ảnh gốc (vật bị cắt), mask phủ >= 35% bbox,
không bị vật khác (bất kỳ lớp nào) che > 15% bbox. Lấy mẫu CÂN BẰNG theo lớp.
Nền: ảnh + nhãn YOLO (vd data/train của BTC); chỉ dán vào chỗ không đè lên box có sẵn.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
# tên lớp COCO 2017 và LVIS v1 -> lớp cuộc thi (LVIS: chỉ dùng vật CÓ nhãn, không dùng cả ảnh vì nhãn không đủ)
COCO2OURS = {
    # COCO
    "bottle": "bottle", "cup": "cup", "wine glass": "cup", "cell phone": "phone", "book": "book", "mouse": "mouse",
    # LVIS
    "water_bottle": "bottle", "wine_bottle": "bottle", "beer_bottle": "bottle", "thermos_bottle": "bottle",
    "glass_(drink_container)": "cup", "mug": "cup", "Dixie_cup": "cup", "teacup": "cup", "wineglass": "cup",
    "cellular_telephone": "phone", "notebook": "book", "booklet": "book", "paperback_book": "book",
    "mouse_(computer_equipment)": "mouse",
    "box": "box", "carton": "box", "matchbox": "box", "pencil_box": "box", "cigarette_case": "box",
}
SIZE_CM = {"bottle": (7, 25), "cup": (7, 12), "phone": (13, 17), "book": (14, 30), "mouse": (9, 13),
           "box": (10, 30), "toy_car": (5, 13)}
W, H = 320, 240


def ioa(a, b):
    """phần của a bị b che (a, b = x1, y1, x2, y2)"""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if iw <= 0 or ih <= 0 else iw * ih / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def img_rel(im):
    """đường dẫn ảnh tương đối thư mục gốc COCO: LVIS không có file_name, chỉ có coco_url .../val2017/xxx.jpg"""
    if "coco_url" in im:
        return "/".join(im["coco_url"].split("/")[-2:])
    return im["file_name"]


def select(ann_path, min_px, pool=None):
    print(f"đọc {ann_path} ...", flush=True)
    d = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    del d
    pool = defaultdict(list) if pool is None else pool
    for iid, anns in by_img.items():
        im = imgs[iid]
        for a in anns:
            name = COCO2OURS.get(cats[a["category_id"]])
            if not name or a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            if min(w, h) < min_px or x < 2 or y < 2 or x + w > im["width"] - 2 or y + h > im["height"] - 2:
                continue
            if a["area"] < 0.35 * w * h:
                continue
            bb = (x, y, x + w, y + h)
            if any(ioa(bb, (o["bbox"][0], o["bbox"][1], o["bbox"][0] + o["bbox"][2], o["bbox"][1] + o["bbox"][3])) > 0.15
                   for o in anns if o is not a):
                continue
            pool[name].append((img_rel(im), im["height"], im["width"], a["segmentation"], bb))
    return pool


def load_bg(bg_dir):
    out = []
    for lp in sorted((Path(bg_dir) / "labels").glob("*.txt")):
        for ext in (".png", ".jpg", ".jpeg"):
            ip = Path(bg_dir) / "images" / (lp.stem + ext)
            if ip.exists():
                out.append((ip, lp))
                break
    return out


def cutout(images_dir, item, rng):
    if item[0] == "CROP":   # vật cắt hình chữ nhật (Open Images / Hot Wheels); có <tên>_m.png (GrabCut) thì dùng mask đó
        im = cv2.imread(item[1])
        if im is None:
            return None
        mp = item[1].rsplit(".", 1)[0] + "_m.png"
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if Path(mp).exists() else None
        return im, (m > 127).astype(np.uint8) if m is not None and m.shape == im.shape[:2] else np.ones(im.shape[:2], np.uint8)
    from pycocotools import mask as mu
    fn, ih, iw, seg, (x1, y1, x2, y2) = item
    im = cv2.imread(str(Path(images_dir) / fn))
    if im is None:
        return None
    rles = mu.frPyObjects(seg, ih, iw) if isinstance(seg, list) else [seg]
    m = mu.decode(mu.merge(rles) if isinstance(rles, list) else rles).astype(np.uint8)
    p = 2
    a, b, c, d = max(0, int(x1) - p), max(0, int(y1) - p), min(iw, int(x2) + p + 1), min(ih, int(y2) + p + 1)
    return im[b:d, a:c].copy(), m[b:d, a:c].copy()


def transform(crop, mask, target_px, rng):
    h, w = mask.shape
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    crop = cv2.warpAffine(crop, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    crop, mask = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = target_px / max(crop.shape[:2])
    if s > 1.3:          # vật quá nhỏ trong ảnh gốc: phóng to nhiều thì mờ, bỏ
        return None
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.5:
        crop, mask = cv2.flip(crop, 1), cv2.flip(mask, 1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.75, 1.25)
    crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return crop, mask


def near_pos(o, w, h, rng):
    """vị trí sát cạnh box o (trái/phải/trên/dưới), khe hở -8%..+6 px -> 2 vật cùng lớp nằm sát nhau"""
    x1, y1, x2, y2 = o
    side = rng.randrange(4)
    gx = rng.randint(-int(0.08 * w), 6)
    gy = rng.randint(-int(0.08 * h), 6)
    if side == 0:
        x, y = x2 + gx, rng.randint(int(y1 - h / 2), int(y2 - h / 2))
    elif side == 1:
        x, y = x1 - w - gx, rng.randint(int(y1 - h / 2), int(y2 - h / 2))
    elif side == 2:
        x, y = rng.randint(int(x1 - w / 2), int(x2 - w / 2)), y2 + gy
    else:
        x, y = rng.randint(int(x1 - w / 2), int(x2 - w / 2)), y1 - h - gy
    return int(x), int(y)


def place(canvas, boxes, crop, mask, cls, rng, tries=40, occlude=False, occ=(0.05, 0.15), near=None, edge=False):
    """occlude=False: chỉ đặt vào chỗ trống. occlude=True: vật mới đè lên MỘT PHẦN vật có sẵn (che occ[0]..occ[1]
    diện tích box bên dưới, không vật nào bị che quá occ[1]) -> model học nhận vật bị che khi đồ xếp sát nhau.
    near=box: đặt SÁT CẠNH box đó (chạm / hở vài px, che tối đa ~8%) -> 2 vật cùng lớp sát nhau không bị gộp 1 box.
    edge=True: cho vật LẤN RA NGOÀI mép khung ảnh (còn thấy >= 50%), nhãn = phần thấy được (như chai sát góc ảnh BTC)."""
    h, w = mask.shape
    if h >= H - 2 or w >= W - 2:
        return False
    for _ in range(tries):
        if near is not None:
            x, y = near_pos(near, w, h, rng)
            if x < 1 or y < 1 or x + w > W - 1 or y + h > H - 1:
                continue
        elif edge:
            x, y = rng.randrange(-w // 2, W - w // 2), rng.randrange(-h // 2, H - h // 2)
            vis = (min(W, x + w) - max(0, x)) * (min(H, y + h) - max(0, y)) / (w * h)
            if vis < 0.5 or (0 <= x and x + w <= W and 0 <= y and y + h <= H):   # phải thực sự lấn mép
                continue
        else:
            x, y = rng.randrange(1, W - w - 1), rng.randrange(1, H - h - 1)
        b = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
        if near is not None:
            if any(ioa(b, o[1:]) > 0.08 or ioa(o[1:], b) > 0.08 for o in boxes):
                continue
        elif occlude and boxes:
            cov = [ioa(o[1:], b) for o in boxes]
            if not (occ[0] <= max(cov) and all(c <= occ[1] for c in cov)):
                continue
        elif any(ioa(b, o[1:]) > 0.1 or ioa(o[1:], b) > 0.1 for o in boxes):
            continue
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.8)[..., None]
        # phần nằm trong khung (edge: vật có thể lấn ra ngoài)
        cx0, cy0, cx1, cy1 = max(0, -x), max(0, -y), w - max(0, x + w - W), h - max(0, y + h - H)
        al, cr = alpha[cy0:cy1, cx0:cx1], crop[cy0:cy1, cx0:cx1]
        X0, Y0 = x + cx0, y + cy0
        roi = canvas[Y0:Y0 + al.shape[0], X0:X0 + al.shape[1]].astype(np.float32)
        # bóng đổ nhẹ phía dưới-phải cho vật "nằm" trên bàn
        sh = np.zeros_like(al)
        sh[2:, 2:] = al[:-2, :-2]
        roi *= 1 - 0.25 * sh * (1 - al)
        canvas[Y0:Y0 + al.shape[0], X0:X0 + al.shape[1]] = (al * cr + (1 - al) * roi).astype(np.uint8)
        ys, xs = np.where(mask[cy0:cy1, cx0:cx1] > 0)   # nhãn = phần vật THẤY được
        if len(xs) < 10:
            return False
        boxes.append((cls, X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1))
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", action="append", default=[], help="file nhãn COCO/LVIS (lặp lại được)")
    ap.add_argument("--pool-cache", default="", help="file .pkl kho vật đã lọc: có thì đọc (bỏ qua --ann), chưa có thì ghi")
    ap.add_argument("--prefix", default="coco", help="tiền tố tên ảnh (chạy song song nhiều phần không trùng tên)")
    ap.add_argument("--only-pool", action="store_true", help="chỉ lọc kho vật rồi ghi --pool-cache, không sinh ảnh")
    ap.add_argument("--images", required=True, help="thư mục gốc COCO chứa train2017/ và val2017/")
    ap.add_argument("--bg", required=True, help="thư mục nền images/ + labels/ (YOLO)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--per-img", default="2,5", help="số vật dán mỗi ảnh (min,max)")
    ap.add_argument("--px-per-cm", default="4.5,6.5", help="khoảng px/cm (camera cao/thấp)")
    ap.add_argument("--min-px", type=int, default=40, help="cạnh bbox COCO nhỏ nhất (vật đủ chi tiết)")
    ap.add_argument("--max-px", type=float, default=140)
    ap.add_argument("--occlude", type=float, default=0.2, help="xác suất vật dán đè một phần lên vật có sẵn (~20%% số vật)")
    ap.add_argument("--crops", action="append", default=[],
                    help="thư mục vật cắt hình chữ nhật có index.csv (file,cls,...) - Open Images, Hot Wheels")
    ap.add_argument("--edge-cut", type=float, default=0.15, help="xác suất vật lấn ra ngoài mép khung ảnh (còn >= 50%%)")
    ap.add_argument("--class-weights", default="",
                    help="tỉ lệ chọn lớp khi dán, vd 'book=1.5,mouse=1.5,phone=0.7' (lớp không nêu = 1)")
    ap.add_argument("--same-near", type=float, default=0.25,
                    help="xác suất vật dán CÙNG LỚP với 1 vật có sẵn và đặt sát cạnh (2 điện thoại cạnh nhau...)")
    ap.add_argument("--occ", default="0.05,0.15", help="vật dán đè che bao nhiêu phần box bên dưới (min,max), ~10%%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import pickle
    if a.pool_cache and Path(a.pool_cache).exists():
        pool = pickle.load(open(a.pool_cache, "rb"))
    else:
        pool = None
        for ap_ in a.ann:
            pool = select(ap_, a.min_px, pool)
        if a.pool_cache:
            pickle.dump(dict(pool), open(a.pool_cache, "wb"))
    import csv
    for cdir in a.crops:   # thêm vật cắt hình chữ nhật vào kho
        n0 = sum(len(v) for v in pool.values())
        for row in csv.DictReader(open(Path(cdir) / "index.csv")):
            pool.setdefault(row["cls"], []).append(("CROP", str(Path(cdir) / row["file"])))
        print(f"thêm {sum(len(v) for v in pool.values()) - n0} vật cắt từ {cdir}", flush=True)
    if a.only_pool:
        print("kho vật: " + ", ".join(f"{k} {len(v)}" for k, v in pool.items()), flush=True)
        return
    print("vật COCO/LVIS dùng được: " + ", ".join(f"{k} {len(v)}" for k, v in pool.items()), flush=True)
    bgs = load_bg(a.bg)
    lo, hi = map(int, a.per_img.split(","))
    plo, phi = map(float, a.px_per_cm.split(","))
    occ = tuple(map(float, a.occ.split(",")))
    wmap = {k: float(v) for k, v in (kv.split("=") for kv in a.class_weights.split(",") if kv)}
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    classes = sorted(pool)
    cw = [wmap.get(c, 1.0) for c in classes]
    print("trọng số lớp khi dán: " + ", ".join(f"{c} {w:g}" for c, w in zip(classes, cw)), flush=True)
    counts, tiles, used = defaultdict(int), [], defaultdict(set)
    for k in range(a.n):
        ip, lp = rng.choice(bgs)
        canvas = cv2.resize(cv2.imread(str(ip)), (W, H))
        boxes = []
        for l in lp.read_text().split("\n"):
            if l.strip():
                c, x, y, w, h = map(float, l.split())
                boxes.append((int(c), (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        ppc = rng.uniform(plo, phi)
        for _ in range(rng.randint(lo, hi)):
            name, near = rng.choices(classes, cw)[0], None   # theo trọng số lớp (mặc định đều nhau)
            same = [o for o in boxes if NAMES[o[0]] in pool]
            if same and rng.random() < a.same_near:      # cùng lớp với 1 vật có sẵn, đặt SÁT cạnh nó
                o = rng.choices(same, [wmap.get(NAMES[x[0]], 1.0) for x in same])[0]   # cũng theo trọng số lớp
                name, near = NAMES[o[0]], o[1:]
            for _try in range(3):   # đặt hụt (vật to, hết chỗ) -> thử vật khác CÙNG lớp, để số vật bám đúng trọng số
                j = rng.randrange(len(pool[name]))
                r = cutout(a.images, pool[name][j], rng)
                if r is None:
                    continue
                smin, smax = SIZE_CM[name]
                r = transform(*r, min(a.max_px, rng.uniform(smin, smax) * ppc), rng)
                if r is not None and place(canvas, boxes, *r, NAMES.index(name), rng,
                                           occlude=rng.random() < a.occlude, occ=occ, near=near,
                                           edge=near is None and rng.random() < a.edge_cut):
                    counts[name] += 1
                    used[name].add(j)
                    break
        name = f"{a.prefix}_{k:05d}"
        cv2.imwrite(str(out / "images" / (name + ".jpg")), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / (name + ".txt")).write_text("".join(
            f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
            for c, x1, y1, x2, y2 in boxes))
        if a.preview and len(tiles) < 12:
            t = canvas.copy()
            for c, x1, y1, x2, y2 in boxes:
                cv2.rectangle(t, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(t, NAMES[c], (int(x1) + 1, int(y1) + 9), 0, 0.3, (0, 0, 255), 1)
            tiles.append(t)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{a.n}", flush=True)
    print(f"{a.n} ảnh -> {out} | vật đã dán (số vật KHÁC NHAU, lặp TB): " + ", ".join(
        f"{k} {v} ({len(used[k])}, x{v / max(1, len(used[k])):.1f})" for k, v in counts.items()), flush=True)
    if a.preview and tiles:
        cv2.imwrite(a.preview, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
