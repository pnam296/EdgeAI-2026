"""Thu nho anh Roboflow ve dung ti le camera that (OV7670 o do cao ~70cm) roi ghep mosaic 2x2 -> anh 320x240.

Anh Roboflow phan lon chup can canh (vat 100-300px). Camera that o 70cm chi thay ~5 px/cm, nen coc ~50px,
dien thoai ~80px. Script chon kich thuoc muc tieu theo KICH THUOC THAT cua tung loai vat (cm) x px/cm,
thu nho ca anh cho vat lon nhat dat kich thuoc do, roi ghep 4 anh vao 1 khung 320x240 (khong phong to anh nao).

    python tools/zoom_out.py --src data/rf_merged --out data/rf_zoom
    python tools/zoom_out.py --src data/rf_merged --out data/rf_zoom --px-per-cm 5.0 --ratio 0.5 --preview outputs/zoom.jpg

--ratio: so anh ra / so anh vao (moi anh vao xuat hien ~4*ratio lan, o cac kich thuoc khac nhau).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 320, 240
# canh dai vat (cm) nhin tu camera ~70cm, hoi nghieng. Chai: dung (nhin thay nap+than) -> nam ngang.
SIZE_CM = {"bottle": (7, 25), "cup": (7, 12), "phone": (13, 17), "book": (14, 30),
           "mouse": (9, 13), "box": (12, 32), "toy_car": (5, 13)}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_rgb(p):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def save_rgb(p, im):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1].tofile(str(p))


def read_labels(p, iw, ih):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) == 5:
                c, x, y, w, h = int(v[0]), *map(float, v[1:])
                out.append([c, (x - w / 2) * iw, (y - h / 2) * ih, (x + w / 2) * iw, (y + h / 2) * ih])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thu muc co images/ + labels/ (vd data/rf_merged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="data/classes.txt")
    ap.add_argument("--px-per-cm", type=float, default=5.0, help="do bang to A4 duoi camera that")
    ap.add_argument("--jitter", type=float, default=0.2, help="dao dong them +-ti le (camera cao/thap, nghieng)")
    ap.add_argument("--max-px", type=float, default=140, help="khong de vat lon hon (gioi han box cua model)")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--min-keep", type=float, default=0.95,
                    help="giu box neu con >= ti le dien tich sau khi cat (thap -> vat mat goc canh van co nhan)")
    ap.add_argument("--edge", type=float, default=0.01,
                    help="box cach mep anh GOC < ti le nay = vat bi cat san: vat chinh bi cat -> bo anh, vat phu -> xoa")
    ap.add_argument("--fit", type=float, default=0.9, help="vat lon nhat toi da ti le nay cua o ghep (khong bi cat)")
    ap.add_argument("--min-px", type=float, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    classes = [l.strip() for l in Path(args.classes).read_text(encoding="utf-8").splitlines() if l.strip()]
    src, out = Path(args.src), Path(args.out)
    imgs = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    n_out = max(1, int(len(imgs) * args.ratio))
    pool = []

    def next_img():
        if not pool:
            pool.extend(imgs)
            rng.shuffle(pool)
        return pool.pop()

    skipped = [0]

    def truncated(b, iw, ih):
        m = args.edge
        return b[1] < m * iw or b[2] < m * ih or b[3] > (1 - m) * iw or b[4] > (1 - m) * ih

    def shrunk(p, qw, qh):
        """anh da thu nho + box (pixel, kem co 'bi cat san') theo kich thuoc that cua vat lon nhat, vua o qw x qh.
        None neu vat chinh bi cat san trong anh goc (chup can canh) -> khong dung duoc."""
        im = load_rgb(p)
        if im is None:
            return None
        ih, iw = im.shape[:2]
        boxes = read_labels(src / "labels" / (p.stem + ".txt"), iw, ih)
        if boxes:
            big = max(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
            if truncated(big, iw, ih):
                return None
            bw, bh = big[3] - big[1], big[4] - big[2]
            lo, hi = SIZE_CM.get(classes[big[0]], (8, 20))
            tgt = rng.uniform(lo, hi) * args.px_per_cm * rng.uniform(1 - args.jitter, 1 + args.jitter)
            s = min(1.0, min(tgt, args.max_px) / max(bw, bh, 1), args.fit * qw / max(bw, 1), args.fit * qh / max(bh, 1))
        else:
            s = rng.uniform(0.4, 1.0)
        nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        return im, [[c, x1 * s, y1 * s, x2 * s, y2 * s, truncated([c, x1, y1, x2, y2], iw, ih)]
                    for c, x1, y1, x2, y2 in boxes]

    def next_usable(qw, qh):
        for _ in range(50):
            r = shrunk(next_img(), qw, qh)
            if r is not None:
                return r
            skipped[0] += 1
        raise RuntimeError("qua nhieu anh co vat chinh bi cat san")

    def cells():
        """1, 2 (ngang/doc) hoac 4 o: vat lon (box, sach) van co cho hien du kich thuoc."""
        r = rng.random()
        if r < 0.2:
            return [(0, 0, W, H)]
        if r < 0.45:
            if rng.random() < 0.5:
                cx = rng.randint(int(W * 0.35), int(W * 0.65))
                return [(0, 0, cx, H), (cx, 0, W, H)]
            cy = rng.randint(int(H * 0.35), int(H * 0.65))
            return [(0, 0, W, cy), (0, cy, W, H)]
        cx, cy = rng.randint(int(W * 0.3), int(W * 0.7)), rng.randint(int(H * 0.3), int(H * 0.7))
        return [(0, 0, cx, cy), (cx, 0, W, cy), (0, cy, cx, H), (cx, cy, W, H)]

    sizes, total, erased = [], 0, 0
    previews = []
    for k in range(n_out):
        canvas = np.zeros((H, W, 3), np.uint8)
        labels = []
        for qx0, qy0, qx1, qy1 in cells():
            qw, qh = qx1 - qx0, qy1 - qy0
            im, boxes = next_usable(qw, qh)
            ih, iw = im.shape[:2]
            # nen: mau trung binh vien anh + nhieu nhe (khong lap lai vat chua gan nhan)
            border = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]).mean(0)
            canvas[qy0:qy1, qx0:qx1] = np.clip(border + np.random.default_rng(k).normal(0, 6, (qh, qw, 3)), 0, 255)
            # anh lon hon o -> cat ngau nhien; nho hon -> dat ngau nhien trong o
            # uu tien vi tri cat chua tron cac vat (hop bao tat ca box, khong duoc thi vat lon nhat)
            def pick(n, q, lo, hi):
                if n <= q:
                    return -rng.randint(0, q - n)
                a, b = max(0, int(np.ceil(hi)) - q), min(n - q, int(lo))  # khoang offset giu tron [lo, hi]
                return rng.randint(a, b) if a <= b else rng.randint(0, n - q)

            if boxes:
                big = max(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
                ux1, uy1 = min(b[1] for b in boxes), min(b[2] for b in boxes)
                ux2, uy2 = max(b[3] for b in boxes), max(b[4] for b in boxes)
                if ux2 - ux1 > qw or uy2 - uy1 > qh:
                    ux1, uy1, ux2, uy2 = big[1:5]
            else:
                ux1 = uy1 = ux2 = uy2 = 0
            ox = pick(iw, qw, ux1, ux2)  # toa do anh tai goc o (am = anh nho hon o, dat lui vao)
            oy = pick(ih, qh, uy1, uy2)
            sx0, sy0 = max(ox, 0), max(oy, 0)
            sx1, sy1 = min(ox + qw, iw), min(oy + qh, ih)
            canvas[qy0 + sy0 - oy:qy0 + sy1 - oy, qx0 + sx0 - ox:qx0 + sx1 - ox] = im[sy0:sy1, sx0:sx1]
            for c, x1, y1, x2, y2, cut0 in boxes:
                a0 = (x2 - x1) * (y2 - y1)
                bx1, by1 = max(x1, sx0), max(y1, sy0)
                bx2, by2 = min(x2, sx1), min(y2, sy1)
                if a0 <= 0 or bx2 <= bx1 or by2 <= by1:
                    continue
                X1, Y1 = qx0 + bx1 - ox, qy0 + by1 - oy
                X2, Y2 = qx0 + bx2 - ox, qy0 + by2 - oy
                if cut0 or (bx2 - bx1) * (by2 - by1) / a0 < args.min_keep or bx2 - bx1 < args.min_px \
                        or by2 - by1 < args.min_px:
                    # vat bi cat do: xoa phan con thay bang mau nen de khong thanh vat khong nhan
                    r = canvas[int(Y1):int(np.ceil(Y2)), int(X1):int(np.ceil(X2))]
                    r[:] = np.clip(border + np.random.default_rng(k + 1).normal(0, 6, r.shape), 0, 255)
                    erased += 1
                    continue
                labels.append((c, X1, Y1, X2, Y2))
                sizes.append(max(X2 - X1, Y2 - Y1))
        name = f"zoom_{k:05d}"
        save_rgb(out / "images" / f"{name}.png", canvas)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        (out / "labels" / f"{name}.txt").write_text(
            "".join(f"{c} {(a + b) / 2 / W:.6f} {(d + e) / 2 / H:.6f} {(b - a) / W:.6f} {(e - d) / H:.6f}\n"
                    for c, a, d, b, e in labels), encoding="utf-8")
        total += len(labels)
        if args.preview and len(previews) < 12:
            v = canvas.copy()
            for c, a, d, b, e in labels:
                cv2.rectangle(v, (int(a), int(d)), (int(b), int(e)), (255, 0, 0), 1)
                cv2.putText(v, classes[c], (int(a) + 1, int(d) + 8), 0, 0.3, (255, 255, 0), 1)
            previews.append(v)
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{n_out}", flush=True)

    s = np.array(sizes)
    print(f"{n_out} anh, {total} box | canh dai: median {np.median(s):.0f}px, p90 {np.percentile(s, 90):.0f}, "
          f"max {s.max():.0f} | >96px {np.mean(s > 96) * 100:.0f}% | xoa {erased} vat bi cat | bo {skipped[0]} anh vat chinh bi cat san")
    if previews:
        while len(previews) % 4:
            previews.append(np.zeros((H, W, 3), np.uint8))
        grid = np.vstack([np.hstack(previews[i:i + 4]) for i in range(0, len(previews), 4)])
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        save_rgb(Path(args.preview), cv2.resize(grid, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
        print("preview:", args.preview)


if __name__ == "__main__":
    main()
