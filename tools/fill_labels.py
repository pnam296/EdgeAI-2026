"""Bo sung nhan bi sot trong 1 dataset Roboflow (YOLO) bang model COCO pretrained.

Vd dataset phone: nhieu dien thoai mau trang khong duoc gan nhan -> model hoc "dien thoai trang = nen".
Script chay YOLO COCO tren tung anh, box nao cua class can bo sung ma KHONG trung box co san (IoU < --iou)
thi them vao file nhan. Nhan goc duoc sao luu sang labels_orig/ (chi lan dau).

    python tools/fill_labels.py --src data/Roboflow/phone_v1i_yolov8 --cls phone
    python tools/fill_labels.py --src data/Roboflow/phone_v1i_yolov8 --cls phone --preview outputs/fill_phone.jpg --dry-run
Xem preview: do = nhan goc giu lai, xanh = nhan them, vang gach cheo = nhan goc bi go (nam tren laptop...).

--cls: ten class trong data.yaml cua dataset nguon (vd 'phone'). COCO co: cell phone, cup, bottle, book, mouse
(khong co box, toy car).
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

COCO_OF = {"phone": "cell phone", "cell phone": "cell phone", "cup": "cup", "bottle": "bottle",
           "book": "book", "mouse": "mouse"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_boxes(lp: Path):
    out = []
    if not lp.exists():
        return out
    for line in lp.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, v = int(p[0]), list(map(float, p[1:]))
        if len(v) == 4:
            x, y, w, h = v
            out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        else:  # polygon
            xs, ys = v[0::2], v[1::2]
            out.append((c, min(xs), min(ys), max(xs), max(ys)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thu muc dataset Roboflow (co data.yaml, */images, */labels)")
    ap.add_argument("--cls", required=True, help="ten class trong data.yaml can bo sung")
    ap.add_argument("--model", default="yolo11x.pt", help="model COCO ultralytics (tu tai lan dau)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3, help="trung box co san neu IoU >= gia tri nay")
    ap.add_argument("--min-side", type=float, default=0.02, help="bo box nho hon ti le canh anh nay")
    ap.add_argument("--prune-on", default="laptop,keyboard",
                    help="go nhan goc nam gon trong cac vat COCO nay ma khong co detection class can sua ('' = tat)")
    ap.add_argument("--dry-run",action="store_true", help="chi dem + preview, khong ghi nhan")
    ap.add_argument("--preview", default="", help="ghi anh xem truoc (do = nhan goc, xanh = nhan them)")
    args = ap.parse_args()

    from ultralytics import YOLO

    src = Path(args.src)
    names = yaml.safe_load(next(src.rglob("data.yaml")).read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if args.cls not in names:
        raise SystemExit(f"'{args.cls}' khong co trong data.yaml: {names}")
    cid = names.index(args.cls)
    coco_name = COCO_OF.get(args.cls.lower())
    if coco_name is None:
        raise SystemExit(f"COCO khong co class tuong ung '{args.cls}'")

    model = YOLO(args.model)
    by_name = {v: k for k, v in model.names.items()}
    coco_id = by_name[coco_name]
    conf_ids = [by_name[n] for n in args.prune_on.split(",") if n in by_name]

    images = sorted(p for p in src.glob("*/images/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    n_add, n_drop, n_img_changed, changed = 0, 0, 0, []
    for k in range(0, len(images), 16):
        batch = images[k:k + 16]
        res = model.predict([str(p) for p in batch], conf=0.05, classes=[coco_id] + conf_ids, verbose=False)
        for ip, r in zip(batch, res):
            lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
            old = read_boxes(lp)
            xy = r.boxes.xyxyn.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            mine = [tuple(map(float, b)) for b, c in zip(xy, cl) if c == coco_id]          # moi muc conf
            other = [tuple(map(float, b)) for b, c, s in zip(xy, cl, cf) if c != coco_id and s >= 0.25]

            # go nhan goc sai: khong co detection class nay nao trung, nhung nam gon trong vat khac (laptop...)
            drop = []
            for o in old:
                if o[0] != cid or any(iou(o[1:], m) >= args.iou for m in mine):
                    continue
                a = (o[3] - o[1]) * (o[4] - o[2])
                for q in other:
                    ix = max(0.0, min(o[3], q[2]) - max(o[1], q[0]))
                    iy = max(0.0, min(o[4], q[3]) - max(o[2], q[1]))
                    if a > 0 and ix * iy / a >= 0.6:
                        drop.append(o)
                        break
            old = [o for o in old if o not in drop]

            add = []
            for b, c, s in zip(xy, cl, cf):
                if c != coco_id or s < args.conf:
                    continue
                b = tuple(float(x) for x in b)
                if min(b[2] - b[0], b[3] - b[1]) < args.min_side:
                    continue
                if any(iou(b, o[1:]) >= args.iou for o in old) or any(iou(b, a) >= args.iou for a in add):
                    continue
                add.append(b)
            if not add and not drop:
                continue
            n_add += len(add)
            n_drop += len(drop)
            n_img_changed += 1
            changed.append((ip, old, add, drop))
            if args.dry_run:
                continue
            bak = ip.parent.parent / "labels_orig" / lp.name
            if not bak.exists():
                bak.parent.mkdir(parents=True, exist_ok=True)
                if lp.exists():
                    shutil.copy2(lp, bak)
                else:
                    bak.write_text("", encoding="utf-8")
            lines = [f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                     for c, x1, y1, x2, y2 in old + [(cid, *b) for b in add]]
            lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\r{min(k + 16, len(images))}/{len(images)} anh, them {n_add} box, go {n_drop} box", end="", flush=True)
    print(f"\n{'(dry-run) ' if args.dry_run else ''}them {n_add} box '{args.cls}', go {n_drop} box sai, "
          f"sua {n_img_changed}/{len(images)} anh")

    if args.preview and changed:
        random.seed(0)
        tiles = []
        pick = [x for x in changed if x[3]][:4]  # luon co vai anh bi go nhan
        rest = [x for x in changed if x not in pick]
        pick += random.sample(rest, min(12 - len(pick), len(rest)))
        for ip, old, add, drop in pick:
            im = cv2.imdecode(np.fromfile(str(ip), np.uint8), cv2.IMREAD_COLOR)
            h, w = im.shape[:2]
            t = max(2, w // 200)
            for c, x1, y1, x2, y2 in old:
                cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), t)
            for c, x1, y1, x2, y2 in drop:
                cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 220, 255), t)
                cv2.line(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 220, 255), t)
            for x1, y1, x2, y2 in add:
                cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 220, 0), t)
            tiles.append(cv2.resize(im, (400, 300)))
        while len(tiles) % 4:
            tiles.append(np.zeros((300, 400, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", grid)[1].tofile(args.preview)
        print("preview:", args.preview)


if __name__ == "__main__":
    main()

