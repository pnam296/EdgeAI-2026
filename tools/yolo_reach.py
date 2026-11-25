"""Tang tam voi box cua tang P4 (stride 16) ma KHONG them tham so.

YOLO DFL: moi canh du doan khoang cach 0..(reg_max-1) "don vi" tu tam o; don vi = stride. Voi reg_max 4,
stride lon nhat 16 -> moi phia toi da 3*16 = 48px -> box toi da ~96px.

Module nay doi don vi cua tang cuoi thanh stride*k (k = DIST_SCALE[-1], mac dinh 1.5 -> 72px/phia -> ~144px).
Cach lam: thay make_anchors (dung chung cho head Detect va loss) de voi tang do tra ve
    anchor_points / k   va   stride_tensor * k
-> tam anchor tinh ra pixel van dung ((i+0.5)/k * 16k = (i+0.5)*16), con khoang cach du doan nhan voi 16k.
Loss, assigner, decode luc infer deu tu khop vi deu di qua make_anchors.

Dung: import + apply() TRUOC khi train / predict / val (moi tien trinh):
    import sys; sys.path.insert(0, "tools")
    import yolo_reach; yolo_reach.apply((1.0, 1.0, 1.5))
Model train voi apply(...) PHAI dung cung he so khi infer (Python: apply; ESP: YOLO_DIST_SCALE trong
firmware/main/model_config.h; verify_det.py: --dist-scale).
"""
import torch

DIST_SCALE = (1.0, 1.0, 1.0)
_orig = None


def make_anchors_scaled(feats, strides, grid_cell_offset=0.5):
    ap, st = _orig(feats, strides, grid_cell_offset)
    if all(k == 1.0 for k in DIST_SCALE):
        return ap, st
    ap, st = ap.clone(), st.clone()
    i0 = 0
    for i in range(len(feats)):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        k = DIST_SCALE[i] if i < len(DIST_SCALE) else 1.0
        if k != 1.0:
            ap[i0:i0 + h * w] /= k
            st[i0:i0 + h * w] *= k
        i0 += h * w
    return ap, st


def apply(scales=(1.0, 1.0, 1.5)):
    """Bat he so cho tung tang (P2, P3, P4). Goi lai voi (1,1,1) de tro ve mac dinh."""
    global DIST_SCALE, _orig
    import ultralytics.nn.modules.head as head
    import ultralytics.utils.loss as loss
    import ultralytics.utils.tal as tal

    if _orig is None:
        _orig = tal.make_anchors
    DIST_SCALE = tuple(float(k) for k in scales)
    for mod in (head, loss, tal):
        mod.make_anchors = make_anchors_scaled
    return DIST_SCALE


def max_box(reg_max=4, strides=(4, 8, 16)):
    return [2 * (reg_max - 1) * s * k for s, k in zip(strides, DIST_SCALE)]


if __name__ == "__main__":
    # tu kiem tra: tam anchor pixel khong doi, stride tang dung k
    apply((1.0, 1.0, 1.5))
    feats = [torch.zeros(1, 1, 60, 80), torch.zeros(1, 1, 30, 40), torch.zeros(1, 1, 15, 20)]
    a0, s0 = _orig(feats, torch.tensor([4.0, 8.0, 16.0]))
    a1, s1 = make_anchors_scaled(feats, torch.tensor([4.0, 8.0, 16.0]))
    assert torch.allclose(a0 * s0, a1 * s1), "tam anchor pixel bi lech"
    print("OK | box toi da moi tang (px):", max_box())
