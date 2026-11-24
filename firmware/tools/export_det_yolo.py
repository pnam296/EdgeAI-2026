"""Export detector YOLO P2P4 (Ultralytics, .pt) -> TFLite INT8 cho ESP32-S3 -> main/models/det_model.cc/.h

Không dùng onnx2tf: dựng lại đúng kiến trúc bằng Keras (NHWC) rồi chép trọng số đã fuse Conv+BN từ PyTorch,
so khớp số với PyTorch trước khi lượng tử hóa.

Đầu ra model INT8 = 6 tensor thô của head (CHƯA decode), firmware tự decode (main/logic.c: decode_yolo):
    box_p{2,3,4}: 1 x H x W x (4*reg_max)   logit DFL, thứ tự cạnh l, t, r, b
    cls_p{2,3,4}: 1 x H x W x nc            logit lớp (chưa sigmoid)
Tách box/cls và từng mức để mỗi tensor có scale lượng tử riêng (gộp chung 1 tensor như export mặc định của
Ultralytics thì xác suất lớp 0..1 và tọa độ pixel 0..320 chung 1 scale int8 -> mất gần hết độ chính xác score).

Chạy (dùng python của venv dự án, đã có ultralytics + tensorflow):
    D:/edgeai/edgeai/Scripts/python.exe tools/export_det_yolo.py
    D:/edgeai/edgeai/Scripts/python.exe tools/export_det_yolo.py --weights path/to/best.pt --calib path/to/images
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from tfl_utils import FW, MODELS, list_ops, to_cc, write_ops_header  # noqa: E402

ROOT = FW.parent / "edgeai"   # repo dự án (D:/edgeai/edgeai)
H_IN, W_IN = 240, 320


# ---------------------------------------------------------------- PyTorch -> Keras
def build_keras(det_model):
    import tensorflow as tf
    from ultralytics.nn.modules import SPPF, C2f, C3, Bottleneck, Concat, Conv, Detect, SCDown

    L = tf.keras.layers
    counter = {"n": 0}

    def name(p):
        counter["n"] += 1
        return f"{p}_{counter['n']}"

    def conv2d(x, c: nn.Conv2d, act: bool):
        w = c.weight.detach().numpy()
        b = c.bias.detach().numpy() if c.bias is not None else np.zeros(c.out_channels, np.float32)
        k, s, p, g = c.kernel_size[0], c.stride[0], c.padding[0], c.groups
        assert c.kernel_size[0] == c.kernel_size[1] and c.stride[0] == c.stride[1] and c.dilation == (1, 1)
        if s == 1 and p == k // 2:
            pad = "same"
        else:
            if p:
                x = L.ZeroPadding2D(p, name=name("pad"))(x)
            pad = "valid"
        if g == 1:
            layer = L.Conv2D(c.out_channels, k, s, pad, name=name("conv"))
            x = layer(x)
            layer.set_weights([w.transpose(2, 3, 1, 0), b])
        elif g == c.in_channels == c.out_channels:
            layer = L.DepthwiseConv2D(k, s, pad, name=name("dw"))
            x = layer(x)
            layer.set_weights([w.transpose(2, 3, 0, 1), b])
        else:
            raise NotImplementedError(f"group conv g={g}")
        if act:   # SiLU = x * sigmoid(x) -> LOGISTIC + MUL (TFLite Micro có cả hai)
            x = L.Multiply(name=name("silu"))([x, L.Activation("sigmoid", name=name("sig"))(x)])
        return x

    def conv(x, m):   # ultralytics Conv / DWConv đã fuse (conv có bias, không còn bn)
        assert not hasattr(m, "bn"), "phải fuse() trước"
        if isinstance(m.act, nn.Identity):
            act = False
        elif isinstance(m.act, nn.SiLU):
            act = True
        else:
            raise NotImplementedError(f"activation {m.act}")
        return conv2d(x, m.conv, act)

    def run(m, x):
        if isinstance(m, Conv):                       # gồm DWConv
            return conv(x, m)
        if isinstance(m, nn.Conv2d):
            return conv2d(x, m, act=False)
        if isinstance(m, nn.Sequential):
            for s in m:
                x = run(s, x)
            return x
        if isinstance(m, Bottleneck):
            y = run(m.cv2, run(m.cv1, x))
            return L.Add(name=name("add"))([x, y]) if m.add else y
        if isinstance(m, C2f):                        # gồm C3k2
            a, b = L.Lambda(lambda t: tf.split(t, 2, axis=-1), name=name("split"))(run(m.cv1, x))
            y = [a, b]
            for blk in m.m:
                y.append(run(blk, y[-1]))
            return run(m.cv2, L.Concatenate(name=name("cat"))(y))
        if isinstance(m, C3):                         # gồm C3k
            return run(m.cv3, L.Concatenate(name=name("cat"))([run(m.m, run(m.cv1, x)), run(m.cv2, x)]))
        if isinstance(m, SCDown):
            return run(m.cv2, run(m.cv1, x))
        if isinstance(m, SPPF):
            assert not getattr(m, "add", False)
            y = [run(m.cv1, x)]
            k = m.m.kernel_size
            for _ in range(getattr(m, "n", 3)):
                y.append(L.MaxPooling2D(k, 1, "same", name=name("mp"))(y[-1]))
            return run(m.cv2, L.Concatenate(name=name("cat"))(y))
        if isinstance(m, nn.Upsample):
            assert m.mode == "nearest"
            s = int(m.scale_factor)
            hw = (int(x.shape[1]) * s, int(x.shape[2]) * s)
            # RESIZE_NEAREST_NEIGHBOR (TFLite Micro có); UpSampling2D của Keras 3 sinh TILE (Micro không có)
            return L.Lambda(lambda t: tf.compat.v1.image.resize_nearest_neighbor(
                t, hw, align_corners=False, half_pixel_centers=False), name=name("up"))(x)
        if isinstance(m, nn.Identity):
            return x
        raise NotImplementedError(type(m).__name__)

    inp = tf.keras.Input((H_IN, W_IN, 3), batch_size=1, name="image")
    saved = []
    x = inp
    for m in det_model.model:
        if isinstance(m, Detect):
            assert not getattr(m, "end2end", False), "head end2end chưa hỗ trợ"
            feats = [saved[j] for j in m.f]
            outs = []
            for i, f in enumerate(feats):
                lvl = int(np.log2(float(m.stride[i])))
                outs.append(L.Activation("linear", name=f"box_p{lvl}")(run(m.cv2[i], f)))
                outs.append(L.Activation("linear", name=f"cls_p{lvl}")(run(m.cv3[i], f)))
            return tf.keras.Model(inp, outs, name="yolo_p2p4"), m
        if m.f == -1:
            src = x
        elif isinstance(m.f, int):
            src = saved[m.f]
        else:
            src = [x if j == -1 else saved[j] for j in m.f]
        x = L.Concatenate(name=name("cat"))(src) if isinstance(m, Concat) else run(m, src)
        saved.append(x)
    raise RuntimeError("không thấy Detect")


@torch.no_grad()
def torch_head_outputs(det_model, x):
    """Chạy model PyTorch tới trước Detect rồi trả về [box_p2, cls_p2, ...] dạng NHWC (để so với Keras)."""
    from ultralytics.nn.modules import Detect
    saved, y = [], x
    for m in det_model.model:
        if m.f != -1:
            y = saved[m.f] if isinstance(m.f, int) else [y if j == -1 else saved[j] for j in m.f]
        if isinstance(m, Detect):
            out = []
            for i, f in enumerate(y):
                out += [m.cv2[i](f), m.cv3[i](f)]
            return [o.permute(0, 2, 3, 1).numpy() for o in out]
        y = m(y)
        saved.append(y)


def load_rgb(p):
    im = cv2.imread(str(p))[:, :, ::-1]
    if im.shape[:2] != (H_IN, W_IN):
        im = cv2.resize(im, (W_IN, H_IN), interpolation=cv2.INTER_AREA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "pipeline/weights/yolo_kd_student.pt"))
    ap.add_argument("--calib", default=str(ROOT / "outputs/yolo_tiny/ds_split/images/train"),
                    help="thư mục ảnh để calibrate INT8 (nên trộn thêm ảnh chụp từ OV7670 khi có)")
    ap.add_argument("--calib-extra", default="", help="thư mục ảnh camera (tùy chọn), trộn vào calibrate")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--out", default=str(MODELS))
    ap.add_argument("--name", default="det_model")
    args = ap.parse_args()

    import tensorflow as tf
    from ultralytics import YOLO

    tf.config.set_visible_devices([], "GPU")
    model = YOLO(args.weights).model.float().eval().fuse()
    n_par = sum(p.numel() for p in model.parameters())
    det = model.model[-1]
    print(f"[1/4] {Path(args.weights).name}: {n_par:,} tham số, nc={det.nc}, reg_max={det.reg_max}, "
          f"stride={det.stride.tolist()}")
    assert n_par <= 200_000, "vượt 200k tham số"

    km, det = build_keras(model)
    print(f"      Keras: {km.count_params():,} tham số, đầu ra " + ", ".join(
        f"{o.name.split('/')[0]}{tuple(o.shape[1:])}" for o in km.outputs))

    # So khớp Keras (float) với PyTorch trên ảnh thật
    calib = sorted(Path(args.calib).glob("*.png")) + sorted(Path(args.calib).glob("*.jpg"))
    if args.calib_extra:
        extra = sorted(Path(args.calib_extra).glob("*.jpg")) + sorted(Path(args.calib_extra).glob("*.png"))
        print(f"      + {len(extra)} ảnh camera vào calibrate")
    else:
        extra = []
    assert calib, f"không có ảnh trong {args.calib}"
    x = load_rgb(calib[0]).astype(np.float32)[None] / 255.0
    pt = torch_head_outputs(model, torch.from_numpy(x).permute(0, 3, 1, 2).contiguous())
    kr = [o.numpy() for o in km(x, training=False)]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(pt, kr))
    print(f"[2/4] Keras vs PyTorch: max|diff| = {diff:.2e}")
    assert diff < 1e-3, "Keras lệch PyTorch: kiểm tra lại build_keras"

    # INT8
    rng = random.Random(0)
    pick = rng.sample(calib, min(args.n_calib, len(calib)))
    pick += rng.sample(extra, min(len(extra), args.n_calib // 2)) if extra else []
    rng.shuffle(pick)

    def rep():
        for p in pick:
            yield [load_rgb(p).astype(np.float32)[None] / 255.0]

    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.tflite").write_bytes(tfl)
    print(f"[3/4] INT8: {len(tfl):,} byte ({len(pick)} ảnh calibrate), op: {' '.join(list_ops(tfl))}")

    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i0 = it.get_input_details()[0]
    print(f"      input  {i0['dtype'].__name__} {list(i0['shape'])} q={i0['quantization']}")
    for o in it.get_output_details():
        print(f"      output {o['name']:<28} {list(o['shape'])} q=({o['quantization'][0]:.4f}, {o['quantization'][1]})")
    assert list(i0["shape"]) == [1, H_IN, W_IN, 3] and i0["dtype"] == np.int8

    print("[4/4] Ghi mảng C:")
    to_cc(tfl, args.name, out)
    if out.resolve() == MODELS.resolve():
        write_ops_header()
    print("Xong. Kiểm tra độ chính xác INT8 + decode C: tools/verify_det.py; ước tính thời gian: tools/estimate_latency.py")


if __name__ == "__main__":
    main()
