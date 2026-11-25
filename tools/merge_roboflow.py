"""Gom dataset Roboflow (export định dạng YOLO) về đúng 7 class của cuộc thi, ảnh 320x240.

    # tải trên Roboflow: Download Dataset -> Format "YOLOv8" (hoặc YOLOv5/YOLO11) -> giải nén vào data/roboflow/<tên>
    python tools/merge_roboflow.py --src data/roboflow/cups data/roboflow/bottles --out data/rf_merged
    # tên class lạ: tự ánh xạ thêm (Tên trong data.yaml = class cuộc thi), "-" = bỏ class đó
    python tools/merge_roboflow.py --src data/roboflow/office --out data/rf_merged --map "Coffee Mug=cup,keyboard=-"

Mỗi ảnh: cắt giữa về tỉ lệ 4:3 (giống camera) -> resize 320x240 -> đổi box theo. Box bị cắt mất quá nửa thì bỏ.
Nhãn polygon (export segmentation) được đổi thành box bao ngoài.

CẢNH BÁO quan trọng: dataset chỉ gán nhãn 1-2 class thì các vật class khác trong ảnh KHÔNG có nhãn, model sẽ học
rằng chúng là nền. Script in bảng class mỗi nguồn để bạn kiểm tra; nên dùng model thầy gán bổ sung trước khi train.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from sim_ov7670 import IMG_EXT, save_rgb  # noqa: E402

W, H = 320, 240

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", str(s))).strip().lower()


# tên hay gặp trên Roboflow / COCO -> class cuộc thi
SYNONYMS = {
    "bottle": "bottle", "bottles": "bottle", "water bottle": "bottle", "plastic bottle": "bottle",
    "glass bottle": "bottle", "pet bottle": "bottle",
    "cup": "cup", "cups": "cup", "mug": "cup", "coffee cup": "cup", "paper cup": "cup", "tea cup": "cup",
    "phone": "phone", "cell phone": "phone", "cellphone": "phone", "mobile phone": "phone",
    "smartphone": "phone", "smart phone": "phone", "mobile": "phone", "iphone": "phone",
    "book": "book", "books": "book", "notebook": "book",
    "mouse": "mouse", "computer mouse": "mouse", "mice": "mouse",
    "box": "box", "boxes": "box", "cardboard box": "box", "carton": "box", "carton box": "box", "package": "box",
    "toy car": "toy_car", "toycar": "toy_car", "car toy": "toy_car", "toy cars": "toy_car", "toy vehicle": "toy_car",
}
# đoán theo từ trong tên khi không khớp nguyên cụm (vd "Coffee-Mug" -> cup). Tên chứa từ loại trừ thì không đoán.
KEYWORDS = {"bottle": "bottle", "bottles": "bottle", "cup": "cup", "cups": "cup", "mug": "cup", "mugs": "cup",
            "phone": "phone", "phones": "phone", "smartphone": "phone", "cellphone": "phone", "iphone": "phone",
            "book": "book", "books": "book", "mouse": "mouse", "box": "box", "boxes": "box", "carton": "box",
            "cartons": "box", "cardboard": "box", "parcel": "box", "parcels": "box", "package": "box",
            "packages": "box", "packet": "box", "delivery": "box", "shipment": "box", "crate": "box"}
NOT_KEYWORDS = {"pad", "mousepad", "case", "shelf", "bookshelf", "cap", "lid", "holder", "charger", "cable",
                "solder", "title"}
CAR_WORDS = {"car", "cars", "toycar", "vehicle", "vehicles", "police", "truck", "taxi", "jeep", "suv"}


def guess(name: str) -> str | None:
    n = norm(name)
    if n in SYNONYMS:
        return SYNONYMS[n]
    toks = n.split()
    if NOT_KEYWORDS & set(toks):
        return None
    # moi loai xe (Red Car, Police, Emergency Vehicles...) -> toy_car: vat cuoc thi chi co xe do choi
    if CAR_WORDS & set(toks):
        return "toy_car"
    hits = {KEYWORDS[t] for t in toks if t in KEYWORDS}
    return hits.pop() if len(hits) == 1 else None


def load_classes(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_names(root: Path) -> list[str]:
    ys = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not ys:
        sys.exit(f"{root}: không có data.yaml (cần export Roboflow định dạng YOLO)")
    names = yaml.safe_load(ys[0].read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        sys.exit(f"{ys[0]}: thiếu 'names'")
    return list(names)


def find_pairs(root: Path):
    """(ảnh, nhãn) trong mọi thư mục images/ (train/valid/test hoặc phẳng)."""
    for img_dir in sorted(p for p in root.rglob("images") if p.is_dir()):
        lbl_dir = img_dir.parent / "labels"
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                yield p, lbl_dir / f"{p.stem}.txt"


def parse_label(line: str):
    v = line.split()
    if len(v) < 5:
        return None
    c = int(float(v[0]))
    nums = list(map(float, v[1:]))
    if len(nums) == 4:
        x, y, w, h = nums
        return c, x - w / 2, y - h / 2, x + w / 2, y + h / 2
    xs, ys = nums[0::2], nums[1::2]                 # polygon
    return c, min(xs), min(ys), max(xs), max(ys)


def crop_43(im: np.ndarray):
    """Cắt giữa về 4:3, trả ảnh + (x0, y0, cw, ch) theo pixel ảnh gốc."""
    h, w = im.shape[:2]
    if w * 3 > h * 4:
        cw, ch = h * 4 // 3, h
    else:
        cw, ch = w, w * 3 // 4
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return im[y0:y0 + ch, x0:x0 + cw], (x0, y0, cw, ch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="thư mục export Roboflow (có data.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="data/classes.txt")
    ap.add_argument("--map", default="", help='"Tên gốc=class,Tên khác=-"  (- = bỏ)')
    ap.add_argument("--min-keep", type=float, default=0.5, help="giữ box nếu còn >= tỉ lệ diện tích này sau khi cắt")
    ap.add_argument("--min-px", type=float, default=4.0, help="bỏ box nhỏ hơn số pixel này (ở 320x240)")
    ap.add_argument("--keep-empty", action="store_true", help="giữ ảnh không còn box nào (làm ảnh nền)")
    args = ap.parse_args()

    classes = load_classes(args.classes)
    cid = {c: i for i, c in enumerate(classes)}
    user = {}
    for kv in filter(None, (s.strip() for s in args.map.split(","))):
        k, _, v = kv.partition("=")
        v = v.strip()
        if v != "-" and v not in cid:
            sys.exit(f"--map: '{v}' không phải class cuộc thi {classes}")
        user[norm(k)] = v

    out = Path(args.out)
    report = {}
    total = Counter()
    for src in map(Path, args.src):
        names = read_names(src)
        mapping, how = {}, {}
        for i, n in enumerate(names):
            if norm(n) in user:
                t, how[i] = user[norm(n)], "--map"
            elif norm(n).replace(" ", "_") in cid:
                t, how[i] = norm(n).replace(" ", "_"), ""
            elif norm(n) in SYNONYMS:
                t, how[i] = SYNONYMS[norm(n)], ""
            else:
                t, how[i] = guess(n), "(đoán, kiểm tra lại)"
            mapping[i] = cid.get(t) if t and t != "-" else None
        print(f"\n== {src.name}: {len(names)} class")
        for i, n in enumerate(names):
            tgt = classes[mapping[i]] if mapping[i] is not None else "(bỏ)"
            print(f"   {n!r:28} -> {tgt} {how[i] if mapping[i] is not None or how[i] == '--map' else ''}")

        prefix = re.sub(r"[^A-Za-z0-9]+", "_", src.name).strip("_") or "rf"
        cnt, dropped, n_img, n_empty, n_bad = Counter(), Counter(), 0, 0, 0
        for k, (ip, lp) in enumerate(find_pairs(src)):
            im = cv2.imdecode(np.fromfile(str(ip), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                n_bad += 1
                continue
            ih, iw = im.shape[:2]
            crop, (x0, y0, cw, ch) = crop_43(im)
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
            lines = []
            for line in (lp.read_text(encoding="utf-8").splitlines() if lp.exists() else []):
                b = parse_label(line)
                if b is None:
                    continue
                c, x1, y1, x2, y2 = b
                tgt = mapping.get(c)
                if tgt is None:
                    dropped[names[c] if c < len(names) else c] += 1
                    continue
                # pixel ảnh gốc -> trong vùng cắt
                px1, py1, px2, py2 = x1 * iw - x0, y1 * ih - y0, x2 * iw - x0, y2 * ih - y0
                area0 = max(px2 - px1, 0) * max(py2 - py1, 0)
                cx1, cy1, cx2, cy2 = max(px1, 0), max(py1, 0), min(px2, cw), min(py2, ch)
                area1 = max(cx2 - cx1, 0) * max(cy2 - cy1, 0)
                if area0 <= 0 or area1 / area0 < args.min_keep:
                    dropped["(bị cắt)"] += 1
                    continue
                bw, bh = (cx2 - cx1) / cw, (cy2 - cy1) / ch
                if bw * W < args.min_px or bh * H < args.min_px:
                    dropped["(quá nhỏ)"] += 1
                    continue
                lines.append(f"{tgt} {(cx1 + cx2) / 2 / cw:.6f} {(cy1 + cy2) / 2 / ch:.6f} {bw:.6f} {bh:.6f}")
                cnt[classes[tgt]] += 1
            if not lines and not args.keep_empty:
                n_empty += 1
                continue
            name = f"{prefix}_{k:05d}"
            save_rgb(out / "images" / f"{name}.png", cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            (out / "labels").mkdir(parents=True, exist_ok=True)
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_img += 1
        total.update(cnt)
        report[src.name] = {"images": n_img, "skipped_empty": n_empty, "unreadable": n_bad,
                            "boxes": dict(cnt), "dropped": {str(k): v for k, v in dropped.items()}}
        print(f"   -> {n_img} ảnh, box: {dict(cnt)}")
        if dropped:
            print(f"   bỏ: {dict(dropped)}")
        if n_empty:
            print(f"   {n_empty} ảnh không còn box nào -> bỏ (thêm --keep-empty để giữ làm ảnh nền)")
        missing = [c for c in classes if cnt[c] == 0]
        if missing:
            print(f"   ! nguồn này KHÔNG có nhãn: {missing}. Nếu ảnh có chứa các vật này thì chúng đang bị coi là nền.")

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nTổng box: {dict(total)}\nđã ghi {out / 'images'}, {out / 'labels'}, {out / 'report.json'}")


if __name__ == "__main__":
    main()
