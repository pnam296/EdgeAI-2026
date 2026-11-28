"""Smart hybrid auto-labeler (Method B) for 320x240 real camera images (EdgeAI 2026).

Combines:
  1. COCO pretrained YOLO11x (`yolo11x.pt`, 4-way rotation TTA) for 5 COCO classes:
     bottle, cup (+ wine glass), phone (cell phone), book, mouse; plus COCO
     background distractors (laptop, keyboard, chair, tv, person) to prune false positives.
  2. Open-vocabulary YOLOE (`yoloe-11l-seg.pt`, 4-way rotation TTA) with distractor
     prompts for `toy_car`, `box`, and high-confidence `phone`/`mouse` fallback.
  3. Open-vocabulary YOLO-World (`yolov8l-worldv2.pt`) for extra `box` and `mouse` recall.
  4. High-confidence `toy_car` boost from the competition teacher
     (`pipeline/weights/yolo_teacher.pt`, direct 320x240 inference without padding).

Protects manually edited labels via `--skip-edited-after "YYYY-MM-DD HH:MM:SS"`.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

REPO = Path(__file__).resolve().parents[1]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
COLORS = [
    (0, 140, 255),   # 0 bottle
    (255, 80, 0),    # 1 cup
    (0, 200, 0),     # 2 phone
    (200, 0, 200),   # 3 book
    (0, 210, 210),   # 4 mouse
    (160, 90, 20),   # 5 box
    (0, 0, 255),     # 6 toy_car
]

COCO_TO_OURS = {
    "bottle": 0,
    "cup": 1,
    "wine glass": 1,
    "cell phone": 2,
    "book": 3,
    "mouse": 4,
}
COCO_PRUNE_CLASSES = {"laptop", "keyboard", "chair", "tv"}

YOLOE_PROMPTS = [
    "bottle",
    "cup",
    "cell phone",
    "smartphone",
    "book",
    "computer mouse",
    "cardboard box",
    "carton box",
    "toy car",
    "miniature car",
    "laptop",
    "chair",
    "electric fan",
    "table",
    "cable",
    "person",
]
YOLOE_TO_OURS = {
    "toy car": 6,
    "miniature car": 6,
    "cardboard box": 5,
    "carton box": 5,
    "smartphone": 2,
    "cell phone": 2,
    "computer mouse": 4,
}
YOLOE_PRUNE = {"laptop", "chair", "electric fan"}

WORLD_PROMPTS = [
    "cardboard box",
    "carton box",
    "packaging box",
    "toy car",
    "miniature car",
    "bottle",
    "cup",
    "mug",
    "cell phone",
    "book",
    "computer mouse",
    "laptop",
    "chair",
    "electric fan",
    "cable",
    "person",
    "table",
]
WORLD_TO_OURS = {
    "cardboard box": 5,
    "carton box": 5,
    "packaging box": 5,
    "toy car": 6,
    "miniature car": 6,
    "computer mouse": 4,
}


def unrotate_box_xyxy(box: np.ndarray, rot_name: str) -> np.ndarray:
    """Map normalized [x1, y1, x2, y2] from rotated image back to 0-deg space."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if rot_name == "0":
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    if rot_name == "90cw":
        return np.array([y1, 1.0 - x2, y2, 1.0 - x1], dtype=np.float32)
    if rot_name == "90ccw":
        return np.array([1.0 - y2, x1, 1.0 - y1, x2], dtype=np.float32)
    if rot_name == "180":
        return np.array([1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1], dtype=np.float32)
    raise ValueError(rot_name)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` box area covered by `outer` box."""
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_in = max(0.0, float(inner[2] - inner[0])) * max(0.0, float(inner[3] - inner[1]))
    return inter / area_in if area_in > 0.0 else 0.0


def free_cuda(model) -> None:
    """Release model and GPU cache between stages."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_coco_stage(
    imgs: list[Path],
    weights: str,
    conf: float,
    imgsz: int,
    batch_size: int,
    tta_rot: bool = True,
) -> tuple[dict[Path, list[tuple[int, float, np.ndarray]]], dict[Path, list[np.ndarray]]]:
    """Detect COCO classes (with 4-way rotation TTA) and background prune boxes."""
    from ultralytics import YOLO

    model = YOLO(weights)
    coco_ids = [k for k, v in model.names.items() if v in COCO_TO_OURS or v in COCO_PRUNE_CLASSES]
    preds: dict[Path, list[tuple[int, float, np.ndarray]]] = {p: [] for p in imgs}
    prunes: dict[Path, list[np.ndarray]] = {p: [] for p in imgs}
    rots = (
        [
            ("0", None),
            ("90cw", cv2.ROTATE_90_CLOCKWISE),
            ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ("180", cv2.ROTATE_180),
        ]
        if tta_rot
        else [("0", None)]
    )

    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        raw_imgs, valid_paths = [], []
        for p in chunk:
            im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                raw_imgs.append(im)
                valid_paths.append(p)
        if not raw_imgs:
            continue

        for rot_name, rot_code in rots:
            batch_ims = [im if rot_code is None else cv2.rotate(im, rot_code) for im in raw_imgs]
            res = model.predict(
                batch_ims,
                imgsz=imgsz,
                conf=conf,
                iou=0.5,
                classes=coco_ids,
                verbose=False,
            )
            for p, r in zip(valid_paths, res):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxyn = r.boxes.xyxyn.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for box, s, c in zip(xyxyn, confs, clss):
                    cname = model.names[int(c)]
                    ubox = unrotate_box_xyxy(box, rot_name)
                    if cname in COCO_TO_OURS:
                        preds[p].append((COCO_TO_OURS[cname], float(s), ubox))
                    elif cname in COCO_PRUNE_CLASSES and float(s) >= 0.35 and rot_name == "0":
                        prunes[p].append(ubox)

    free_cuda(model)
    return preds, prunes


def run_yoloe_stage(
    imgs: list[Path],
    weights: str,
    conf_car: float,
    conf_box: float,
    conf_phone: float,
    imgsz: int,
    batch_size: int,
    tta_rot: bool = True,
) -> tuple[dict[Path, list[tuple[int, float, np.ndarray]]], dict[Path, list[np.ndarray]]]:
    """Detect toy_car, box, and fallbacks using YOLOE (with 4-way rotation TTA)."""
    from ultralytics import YOLOE

    model = YOLOE(weights)
    model.set_classes(YOLOE_PROMPTS, model.get_text_pe(YOLOE_PROMPTS))
    base_conf = min(conf_car, conf_box, conf_phone, 0.25)
    preds: dict[Path, list[tuple[int, float, np.ndarray]]] = {p: [] for p in imgs}
    prunes: dict[Path, list[np.ndarray]] = {p: [] for p in imgs}
    rots = (
        [
            ("0", None),
            ("90cw", cv2.ROTATE_90_CLOCKWISE),
            ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ("180", cv2.ROTATE_180),
        ]
        if tta_rot
        else [("0", None)]
    )

    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        raw_imgs, valid_paths = [], []
        for p in chunk:
            im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                raw_imgs.append(im)
                valid_paths.append(p)
        if not raw_imgs:
            continue

        for rot_name, rot_code in rots:
            batch_ims = [im if rot_code is None else cv2.rotate(im, rot_code) for im in raw_imgs]
            res = model.predict(
                batch_ims,
                imgsz=imgsz,
                conf=base_conf,
                iou=0.45,
                agnostic_nms=True,
                verbose=False,
            )
            for p, r in zip(valid_paths, res):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxyn = r.boxes.xyxyn.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for box, s, c in zip(xyxyn, confs, clss):
                    prompt = YOLOE_PROMPTS[int(c)]
                    ubox = unrotate_box_xyxy(box, rot_name)
                    if prompt in YOLOE_PRUNE and float(s) >= 0.40 and rot_name == "0":
                        prunes[p].append(ubox)
                        continue
                    ours = YOLOE_TO_OURS.get(prompt)
                    if ours is None:
                        continue
                    thr = (
                        conf_car
                        if ours == 6
                        else (conf_box if ours == 5 else (conf_phone if ours == 2 else 0.40))
                    )
                    if float(s) >= thr:
                        preds[p].append((ours, float(s), ubox))

    free_cuda(model)
    return preds, prunes


def run_world_stage(
    imgs: list[Path],
    weights: str,
    conf_box: float,
    conf_car: float,
    conf_mouse: float,
    imgsz: int,
    batch_size: int,
) -> dict[Path, list[tuple[int, float, np.ndarray]]]:
    """Detect box, toy_car, and mouse fallback using YOLO-World with distractor prompts."""
    from ultralytics import YOLOWorld

    model = YOLOWorld(weights)
    model.set_classes(WORLD_PROMPTS)
    base_conf = min(conf_box, conf_car, conf_mouse)
    preds: dict[Path, list[tuple[int, float, np.ndarray]]] = {p: [] for p in imgs}

    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        res = model.predict(
            [str(p) for p in chunk],
            imgsz=imgsz,
            conf=base_conf,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )
        for p, r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, s, c in zip(xyxyn, confs, clss):
                prompt = WORLD_PROMPTS[int(c)]
                ours = WORLD_TO_OURS.get(prompt)
                if ours is None:
                    continue
                thr = conf_box if ours == 5 else (conf_car if ours == 6 else conf_mouse)
                if float(s) >= thr:
                    preds[p].append((ours, float(s), box.astype(np.float32)))

    free_cuda(model)
    return preds


@torch.no_grad()
def run_teacher_car_stage(
    imgs: list[Path],
    weights: str,
    conf_car: float,
    batch_size: int,
) -> dict[Path, list[tuple[int, float, np.ndarray]]]:
    """Detect toy_car (class 6) using the 320x240 competition teacher without padding."""
    from ultralytics import YOLO
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights).model.to(device).float().eval()
    preds: dict[Path, list[tuple[int, float, np.ndarray]]] = {p: [] for p in imgs}

    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        tensors, valid_paths = [], []
        for p in chunk:
            im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[:2] != (240, 320):
                im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
            tensors.append(im[:, :, ::-1].transpose(2, 0, 1))
            valid_paths.append(p)
        if not tensors:
            continue
        x = torch.from_numpy(np.ascontiguousarray(np.stack(tensors))).to(device).float() / 255.0
        y = model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        dets = non_max_suppression(y, conf_thres=conf_car, iou_thres=0.5, max_det=50)
        for p, d in zip(valid_paths, dets):
            for x1, y1, x2, y2, s, c in d.cpu().numpy()[:, :6]:
                if int(c) == 6:
                    box = np.array(
                        [
                            np.clip(x1 / 320.0, 0.0, 1.0),
                            np.clip(y1 / 240.0, 0.0, 1.0),
                            np.clip(x2 / 320.0, 0.0, 1.0),
                            np.clip(y2 / 240.0, 0.0, 1.0),
                        ],
                        dtype=np.float32,
                    )
                    preds[p].append((6, float(s), box))

    free_cuda(model)
    return preds


def _is_pruned(cls_id: int, box: np.ndarray, prune_boxes: list[np.ndarray]) -> bool:
    """Drop detections that lie inside background clutter (laptop, chair, fan, person)."""
    if not prune_boxes:
        return False
    for pb in prune_boxes:
        cov = containment_ratio(box, pb)
        # Drop book/phone/box/toy_car caught inside a background laptop, chair, fan, or person
        if cls_id in (2, 3, 5, 6) and (cov >= 0.55 or iou_xyxy(box, pb) >= 0.45):
            return True
    return False


def merge_detections(
    coco_dets: list[tuple[int, float, np.ndarray]],
    yoloe_dets: list[tuple[int, float, np.ndarray]],
    world_dets: list[tuple[int, float, np.ndarray]],
    car_dets: list[tuple[int, float, np.ndarray]],
    prune_boxes: list[np.ndarray],
    nms_iou: float,
    min_box_px: float = 6.0,
) -> list[tuple[int, float, np.ndarray]]:
    """Merge detections across stages, prune background distractors, and deduplicate."""
    all_dets: list[tuple[int, float, np.ndarray]] = []

    # 1. COCO detections (bottle, cup, phone, book, mouse), pruned against background distractors
    coco_clean = [d for d in coco_dets if not _is_pruned(d[0], d[2], prune_boxes)]
    if coco_clean:
        boxes_t = torch.tensor(np.array([d[2] for d in coco_clean]), dtype=torch.float32)
        scores_t = torch.tensor([d[1] for d in coco_clean], dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_t, scores_t, nms_iou).tolist()
        all_dets.extend(coco_clean[k] for k in keep)

    # 2. toy_car from YOLOE + teacher + YOLO-World (deduplicated with NMS)
    car_candidates = [
        d
        for d in (
            [x for x in yoloe_dets if x[0] == 6]
            + car_dets
            + [x for x in world_dets if x[0] == 6]
        )
        if not _is_pruned(6, d[2], prune_boxes)
    ]
    if car_candidates:
        boxes_t = torch.tensor(np.array([d[2] for d in car_candidates]), dtype=torch.float32)
        scores_t = torch.tensor([d[1] for d in car_candidates], dtype=torch.float32)
        for k in torchvision.ops.nms(boxes_t, scores_t, 0.45).tolist():
            c, s, b = car_candidates[k]
            ov = [i for i, (_, _, eb) in enumerate(all_dets) if iou_xyxy(b, eb) >= 0.45]
            if not ov:
                all_dets.append((c, s, b))
            elif all(all_dets[i][0] in (2, 3) and s >= all_dets[i][1] + 0.10 for i in ov):
                for i in reversed(ov):
                    all_dets.pop(i)
                all_dets.append((c, s, b))

    # 3. phone / mouse fallbacks from YOLOE + YOLO-World
    extra_pm = sorted(
        (d for d in (yoloe_dets + world_dets) if d[0] in (2, 4) and not _is_pruned(d[0], d[2], prune_boxes)),
        key=lambda d: -d[1],
    )
    for c, s, b in extra_pm:
        if not any(iou_xyxy(b, eb) >= 0.40 for _, _, eb in all_dets):
            all_dets.append((c, s, b))

    # 4. box from YOLO-World + YOLOE (deduplicated)
    box_candidates = [
        d
        for d in (
            [x for x in world_dets if x[0] == 5]
            + [x for x in yoloe_dets if x[0] == 5]
        )
        if not _is_pruned(5, d[2], prune_boxes)
    ]
    if box_candidates:
        boxes_t = torch.tensor(np.array([d[2] for d in box_candidates]), dtype=torch.float32)
        scores_t = torch.tensor([d[1] for d in box_candidates], dtype=torch.float32)
        for k in torchvision.ops.nms(boxes_t, scores_t, 0.45).tolist():
            c, s, b = box_candidates[k]
            if not any(iou_xyxy(b, eb) >= 0.35 for _, _, eb in all_dets):
                all_dets.append((c, s, b))

    filtered: list[tuple[int, float, np.ndarray]] = []
    for c, s, b in all_dets:
        bw = (b[2] - b[0]) * 320.0
        bh = (b[3] - b[1]) * 240.0
        if bw >= min_box_px and bh >= min_box_px:
            filtered.append((c, s, b))
    return filtered


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Dataset folder (contains images/ or direct image files)")
    ap.add_argument("--coco-weights", default="yolo11x.pt")
    ap.add_argument("--yoloe-weights", default="yoloe-11l-seg.pt")
    ap.add_argument("--world-weights", default="yolov8l-worldv2.pt")
    ap.add_argument("--teacher-weights", default=str(REPO / "pipeline/weights/yolo_teacher.pt"))
    ap.add_argument("--conf-coco", type=float, default=0.32, help="Confidence threshold for COCO classes")
    ap.add_argument("--conf-yoloe-car", type=float, default=0.35, help="Confidence threshold for YOLOE toy_car")
    ap.add_argument("--conf-yoloe-box", type=float, default=0.35, help="Confidence threshold for YOLOE box")
    ap.add_argument("--conf-yoloe-phone", type=float, default=0.32, help="Confidence threshold for YOLOE phone")
    ap.add_argument("--conf-box", type=float, default=0.25, help="Confidence threshold for YOLO-World box")
    ap.add_argument("--conf-world-car", type=float, default=0.30, help="Confidence threshold for YOLO-World toy_car")
    ap.add_argument("--conf-world-mouse", type=float, default=0.40, help="Confidence threshold for YOLO-World mouse")
    ap.add_argument("--conf-teacher-car", type=float, default=0.75, help="Confidence threshold for teacher toy_car")
    ap.add_argument("--no-yoloe", action="store_true", help="Disable YOLOE stage")
    ap.add_argument("--no-teacher-car", action="store_true", help="Disable teacher toy_car boost")
    ap.add_argument("--no-rot-tta", action="store_true", help="Disable 4-way rotation TTA")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference resolution for COCO, YOLOE, and YOLO-World")
    ap.add_argument("--iou", type=float, default=0.55, help="Class-agnostic NMS IoU threshold")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    ap.add_argument("--step", type=int, default=1, help="Subsample stride (e.g. --step 67 with --limit 16)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt labels")
    ap.add_argument(
        "--skip-edited-after",
        default="",
        help="Do not overwrite .txt modified after 'YYYY-MM-DD HH:MM:SS' (protects manual edits)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write .txt labels (preview/stats only)")
    ap.add_argument("--preview", default="", help="Save 4x4 preview grid image")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    names = [l.strip() for l in (REPO / "data/classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    skip_ts = None
    if args.skip_edited_after:
        skip_ts = datetime.datetime.strptime(args.skip_edited_after, "%Y-%m-%d %H:%M:%S").timestamp()

    src = Path(args.src)
    img_dir = src / "images"
    lbl_dir = src / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shutil.move(str(p), img_dir / p.name)

    if not args.dry_run:
        for d in (src, lbl_dir):
            (d / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    imgs = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.step > 1:
        imgs = imgs[:: args.step]
    if args.limit > 0:
        imgs = imgs[: args.limit]

    target_imgs: list[Path] = []
    skipped_manual = 0
    for p in imgs:
        lp = lbl_dir / f"{p.stem}.txt"
        if not args.overwrite and not args.dry_run and lp.exists():
            continue
        if skip_ts is not None and not args.dry_run and lp.exists() and lp.stat().st_mtime > skip_ts:
            skipped_manual += 1
            continue
        target_imgs.append(p)

    if not target_imgs:
        print(f"Khong co anh nao can gan nhan (da giu nguyen {skipped_manual} file sua tay).")
        return

    print(f"Running Stage 1 (COCO {args.coco_weights}) on {len(target_imgs)} images...", flush=True)
    coco_preds, coco_prunes = run_coco_stage(
        target_imgs, args.coco_weights, args.conf_coco, args.imgsz, args.batch, tta_rot=not args.no_rot_tta
    )

    if not args.no_yoloe:
        print(f"Running Stage 2 (YOLOE {args.yoloe_weights}) on {len(target_imgs)} images...", flush=True)
        yoloe_preds, yoloe_prunes = run_yoloe_stage(
            target_imgs,
            args.yoloe_weights,
            args.conf_yoloe_car,
            args.conf_yoloe_box,
            args.conf_yoloe_phone,
            args.imgsz,
            args.batch,
            tta_rot=not args.no_rot_tta,
        )
    else:
        yoloe_preds = {p: [] for p in target_imgs}
        yoloe_prunes = {p: [] for p in target_imgs}

    print(f"Running Stage 3 (YOLO-World {args.world_weights}) on {len(target_imgs)} images...", flush=True)
    world_preds = run_world_stage(
        target_imgs,
        args.world_weights,
        args.conf_box,
        args.conf_world_car,
        args.conf_world_mouse,
        args.imgsz,
        args.batch,
    )

    if not args.no_teacher_car and Path(args.teacher_weights).is_file():
        print(f"Running Stage 4 (Teacher toy_car {args.teacher_weights}) on {len(target_imgs)} images...", flush=True)
        car_preds = run_teacher_car_stage(target_imgs, args.teacher_weights, args.conf_teacher_car, args.batch)
    else:
        car_preds = {p: [] for p in target_imgs}

    counts = np.zeros(len(names), dtype=int)
    n_non_empty = 0
    written = 0
    tiles: list[np.ndarray] = []

    for p in target_imgs:
        lp = lbl_dir / f"{p.stem}.txt"
        # Re-check right before writing so live manual edits in labelImg are never overwritten
        if skip_ts is not None and not args.dry_run and lp.exists() and lp.stat().st_mtime > skip_ts:
            skipped_manual += 1
            continue

        prune_boxes = coco_prunes[p] + yoloe_prunes[p]
        merged = merge_detections(
            coco_preds[p], yoloe_preds[p], world_preds[p], car_preds[p], prune_boxes, args.iou
        )
        lines = []
        for c, s, b in merged:
            x1, y1, x2, y2 = (float(v) for v in b)
            xc = np.clip((x1 + x2) / 2.0, 0.0, 1.0)
            yc = np.clip((y1 + y2) / 2.0, 0.0, 1.0)
            bw = np.clip(x2 - x1, 0.0, 1.0)
            bh = np.clip(y2 - y1, 0.0, 1.0)
            if bw <= 1e-4 or bh <= 1e-4:
                continue
            lines.append(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            counts[c] += 1

        if lines:
            n_non_empty += 1
        if not args.dry_run:
            lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1

        if args.preview and len(tiles) < 16:
            im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                im = cv2.resize(im, (320, 240))
                for c, s, b in merged:
                    px1, py1, px2, py2 = int(b[0] * 320), int(b[1] * 240), int(b[2] * 320), int(b[3] * 240)
                    cv2.rectangle(im, (px1, py1), (px2, py2), COLORS[c], 1)
                    cv2.putText(
                        im,
                        f"{names[c]} {s:.2f}",
                        (px1, max(10, py1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        COLORS[c],
                        1,
                    )
                tiles.append(im)

    mode = "[DRY-RUN] " if args.dry_run else ""
    print(
        f"{mode}{len(target_imgs)} anh | da ghi: {written} | giu nguyen sua tay: {skipped_manual} "
        f"| co box: {n_non_empty} | rong: {len(target_imgs) - n_non_empty}"
    )
    print("So box: " + "  ".join(f"{n} {k}" for n, k in zip(names, counts)))

    if args.preview and tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
        out_p = Path(args.preview)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", grid)[1].tofile(str(out_p))
        print("Preview:", out_p)


if __name__ == "__main__":
    main()
