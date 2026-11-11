"""mAP@0.5 theo đúng cách BTC chấm: nội suy 101 điểm như COCO, trung bình trên các lớp
CÓ trong ground truth. Dùng để tự chấm trên tập val thay vì tốn lượt nộp Kaggle.

Định dạng:
  preds = {"img_id": [{"class_id": c, "score": s, "bbox": [x1, y1, x2, y2]}, ...]}
  gts   = {"img_id": [(class_id, [x1, y1, x2, y2]), ...]}
"""
import numpy as np


def iou(box, boxes):
    """IoU giữa 1 box và mảng N box, định dạng [x1, y1, x2, y2]."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def average_precision(tp, n_gt):
    """tp: mảng 0/1 theo thứ tự score giảm dần. Nội suy 101 điểm recall 0, 0.01, ..., 1."""
    if n_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, dtype=float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    # Precision "bao" từ phải sang: tại mỗi recall lấy precision lớn nhất phía sau
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return ap / 101


def map50(preds, gts, num_classes, iou_thr=0.5, return_per_class=False):
    per_class = {}
    for c in range(num_classes):
        gt_c = {k: np.array([b for cc, b in v if cc == c], dtype=float).reshape(-1, 4) for k, v in gts.items()}
        n_gt = sum(len(v) for v in gt_c.values())
        if n_gt == 0:
            continue  # lớp không có trong GT -> không tính vào trung bình
        dets = [(k, p["score"], p["bbox"]) for k, ps in preds.items() if k in gts
                for p in ps if p["class_id"] == c]
        dets.sort(key=lambda d: -d[1])
        used = {k: np.zeros(len(v), bool) for k, v in gt_c.items()}
        tp = []
        for k, _, box in dets:
            g = gt_c.get(k)
            if g is None or len(g) == 0:
                tp.append(0)
                continue
            ious = iou(np.asarray(box, float), g)
            ious[used[k]] = -1  # mỗi GT chỉ được khớp 1 lần
            j = int(ious.argmax())
            if ious[j] >= iou_thr:
                used[k][j] = True
                tp.append(1)
            else:
                tp.append(0)
        per_class[c] = average_precision(tp, n_gt)
    m = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return (m, per_class) if return_per_class else m
