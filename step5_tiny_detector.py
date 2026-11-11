"""BƯỚC 5 - Detector tí hon ĐÚNG RÀNG BUỘC ESP32-S3 (dùng mô hình này để nộp bài).

Chạy:  python step5_tiny_detector.py          (CPU, khoảng 5-15 phút)

Làm lần lượt:
  1. Train trên 80% ảnh, chấm mAP@0.5 trên 20% ảnh val (cùng tập val với YOLO baseline)
  2. Xuất TensorFlow Lite INT8 và chấm lại mAP bằng chính file INT8 (thứ chạy trên thiết bị)
  3. Kiểm tra ràng buộc: kích thước đầu vào, số tham số, dung lượng, op TFLite Micro
  4. Dự đoán tập test BẰNG MÔ HÌNH INT8 -> outputs/det_predictions.json -> chạy step4 để nộp
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import time

import numpy as np
import tensorflow as tf

from common import OUT, TEST_IMG, TRAIN_LBL, list_images, load_classes, split_train_val
from det_tiny_data import INPUT_H, INPUT_W, WH_LOG, decode, gt_pixels, load_samples, preprocess
from det_tiny_model import DetDataset, build_model, focal_loss, make_box_loss
from metrics import map50

SEED = 42
# THỂ LỆ VÒNG LOẠI: mô hình pretrained CHỈ được dùng ở vòng chung kết -> train từ đầu.
PRETRAINED = False
EPOCHS = 200
BATCH = 16
LR = 2e-3
EVAL_EVERY = 5
RETRAIN = True        # False: bỏ qua train, nạp outputs/tiny_det_best.weights.h5 (chỉ xuất/kiểm tra lại)
TRAIN_ON_ALL = False  # True: gộp cả val vào train (khi đã chốt cấu hình). Lúc đó mAP val không còn ý nghĩa.

# Ràng buộc mục 7 của thể lệ
MAX_INPUT = 160
MAX_PARAMS = 200_000
MAX_BYTES = 1.5 * 1024 * 1024
TFLM_OPS = {  # các op có trong TensorFlow Lite Micro (bản INT8)
    "CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "PAD", "RESIZE_NEAREST_NEIGHBOR", "LOGISTIC",
    "QUANTIZE", "DEQUANTIZE", "CONCATENATION", "RESHAPE", "MEAN", "MUL", "SUB",
    "FULLY_CONNECTED", "MAX_POOL_2D", "AVERAGE_POOL_2D", "RELU", "RELU6", "SOFTMAX", "GATHER",
}

tf.keras.utils.set_random_seed(SEED)
classes = load_classes()
C = len(classes)

# ---------------- 1. Dữ liệu ----------------
train_paths, val_paths = split_train_val(seed=SEED)
if TRAIN_ON_ALL:
    train_paths = train_paths + val_paths
print("Đang nạp ảnh...")
train_s = load_samples(train_paths, TRAIN_LBL)
val_s = load_samples(val_paths, TRAIN_LBL)
test_paths = list_images(TEST_IMG)
print(f"Train {len(train_s)} | Val {len(val_s)} | Test {len(test_paths)} | đầu vào {INPUT_W}x{INPUT_H}")

X_val = np.stack([preprocess(s["img"]) for s in val_s])
gts_val = {s["id"]: gt_pixels(s) for s in val_s}


def evaluate(predict_fn, samples, X, iou_thr=0.5):
    """predict_fn(X) -> (heat, box). Trả về (mAP, mAP từng lớp)."""
    heat, box = predict_fn(X)
    preds = {s["id"]: decode(heat[i], box[i], s["orig_size"]) for i, s in enumerate(samples)}
    return map50(preds, {s["id"]: gt_pixels(s) for s in samples}, C, iou_thr=iou_thr, return_per_class=True)


class MapCheckpoint(tf.keras.callbacks.Callback):
    """Mỗi vài epoch chấm mAP val thật sự (không phải loss) và giữ lại trọng số tốt nhất."""

    def __init__(self, path):
        super().__init__()
        self.path, self.best, self.best_epoch = path, -1.0, 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % EVAL_EVERY and epoch + 1 != EPOCHS:
            return
        m, _ = evaluate(lambda x: self.model.predict(x, verbose=0), val_s, X_val)
        mark = ""
        if m >= self.best:
            self.best, self.best_epoch, mark = m, epoch + 1, "  <- tốt nhất, đã lưu"
            self.model.save_weights(self.path)
        print(f"  epoch {epoch + 1:>3}: loss {logs['loss']:.3f} | val mAP50 {m:.3f}{mark}")


# ---------------- 2. Train ----------------
model = build_model(C, pretrained=PRETRAINED)
print(f"Trọng số khởi tạo: {'ImageNet pretrained' if PRETRAINED else 'ngẫu nhiên (train từ đầu)'}")
n_params = model.count_params()
print(f"Số tham số: {n_params:,} (giới hạn {MAX_PARAMS:,})")
steps = int(np.ceil(len(train_s) / BATCH)) * EPOCHS
model.compile(
    optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(LR, steps, alpha=0.05)),
    loss=[focal_loss, make_box_loss(C)],
)
ckpt = MapCheckpoint(str(OUT / "tiny_det_best.weights.h5"))
if RETRAIN or not os.path.exists(ckpt.path):
    t0 = time.time()
    model.fit(DetDataset(train_s, C, BATCH, train=True, seed=SEED), epochs=EPOCHS, verbose=0, callbacks=[ckpt])
    print(f"Train xong sau {(time.time() - t0) / 60:.1f} phút. Tốt nhất: epoch {ckpt.best_epoch}, val mAP50 {ckpt.best:.3f}")
else:
    print(f"RETRAIN = False -> dùng lại trọng số đã lưu: {ckpt.path}")
model.load_weights(ckpt.path)
model.save(OUT / "tiny_det.keras")

m_float, per_cls = evaluate(lambda x: model.predict(x, verbose=0), val_s, X_val)
_, per_cls30 = evaluate(lambda x: model.predict(x, verbose=0), val_s, X_val, iou_thr=0.3)
# CHẨN ĐOÁN: IoU 0.3 "dễ tính" hơn về vị trí. Nếu AP@0.3 cao hơn hẳn AP@0.5 -> mô hình THẤY vật
# nhưng box lệch (sửa phần box). Nếu cả hai đều thấp -> mô hình KHÔNG THẤY / nhầm lớp (sửa phần heatmap/dữ liệu).
print(f"\nmAP50 val (float32): {m_float:.3f}")
print(f"  {'lớp':<24}{'AP@0.5':>8}{'AP@0.3':>8}  chẩn đoán")
for c, ap in per_cls.items():
    ap30 = per_cls30[c]
    hint = "box lệch" if ap30 - ap > 0.15 else ("khó thấy / nhầm lớp" if ap30 < 0.5 else "")
    print(f"  {classes[c]:<24}{ap:>8.3f}{ap30:>8.3f}  {hint}")

# ---------------- 3. Xuất TFLite INT8 ----------------
def representative_data():
    """Ảnh mẫu để bộ chuyển đổi đo dải giá trị của từng lớp -> chọn thang lượng tử hóa."""
    for s in train_s[:100]:
        yield [preprocess(s["img"])[None]]


# Cố định batch = 1 (đúng như trên thiết bị) -> bộ chuyển đổi tính sẵn mọi kích thước,
# không sinh op SHAPE / STRIDED_SLICE mà TFLite Micro không cần.
# (Bọc bằng một Input batch_size=1 rồi dùng from_keras_model: trọng số được gói thành hằng số.
#  Cách from_concrete_functions để lại READ_VARIABLE và làm hỏng bước lượng tử hóa.)
export_in = tf.keras.Input((INPUT_H, INPUT_W, 3), batch_size=1, name="image")
export_model = tf.keras.Model(export_in, model(export_in, training=False), name="tiny_det_export")
conv = tf.lite.TFLiteConverter.from_keras_model(export_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
tfl_path = OUT / "tiny_det_int8.tflite"
tfl_path.write_bytes(tfl)

# Tắt delegate XNNPACK: chạy bằng kernel chuẩn (gần với TFLite Micro hơn) và liệt kê op thật,
# không bị gộp thành một op "DELEGATE".
interp = tf.lite.Interpreter(
    model_content=tfl,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outs = interp.get_output_details()
heat_d = next(o for o in outs if o["shape"][-1] == C)
box_d = next(o for o in outs if o["shape"][-1] == 4 * C)
in_scale, in_zp = inp["quantization"]


def tflite_predict(X):
    heats, boxes = [], []
    for x in X:
        q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        for d, store in ((heat_d, heats), (box_d, boxes)):
            s, z = d["quantization"]
            store.append((interp.get_tensor(d["index"])[0].astype(np.float32) - z) * s)
    return np.stack(heats), np.stack(boxes)


m_int8, _ = evaluate(tflite_predict, val_s, X_val)

# ---------------- 4. Kiểm tra ràng buộc ----------------
try:
    ops = sorted({o["op_name"] for o in interp._get_ops_details()})
except Exception:
    ops = []
bad_ops = [o for o in ops if o not in TFLM_OPS]
in_shape = tuple(inp["shape"][1:3])
checks = [
    (f"Đầu vào {in_shape[1]}x{in_shape[0]} (mỗi chiều <= {MAX_INPUT})", max(in_shape) <= MAX_INPUT),
    (f"Số tham số {n_params:,} (<= {MAX_PARAMS:,})", n_params <= MAX_PARAMS),
    (f"Dung lượng {len(tfl) / 1024:.0f} KB (<= 1536 KB)", len(tfl) <= MAX_BYTES),
    (f"Kiểu dữ liệu INT8 (vào {inp['dtype'].__name__}, ra {heat_d['dtype'].__name__})",
     inp["dtype"] == np.int8 and heat_d["dtype"] == np.int8),
    (f"Op TFLite Micro: {', '.join(ops) or '(không đọc được)'}", not bad_ops),
]
print("\n===== KIỂM TRA RÀNG BUỘC =====")
for text, ok in checks:
    print(f"  [{'OK' if ok else 'LỖI'}] {text}")
if bad_ops:
    print(f"  Op không chắc có trong TFLite Micro: {bad_ops}")
print(f"\nmAP50 val: float32 {m_float:.3f} -> INT8 {m_int8:.3f} (mất {m_float - m_int8:+.3f} do lượng tử hóa)")
print(f"Ước lượng điểm detection: 0,70 x {m_int8 * 100:.1f} = {0.7 * m_int8 * 100:.1f}")

# Thông số lượng tử hóa cần cho code C trên ESP32
(OUT / "tiny_det_quant.json").write_text(json.dumps({
    "input_w": INPUT_W, "input_h": INPUT_H, "num_classes": C, "classes": classes,
    "input_scale": float(in_scale), "input_zero_point": int(in_zp),
    "heat_scale": float(heat_d["quantization"][0]), "heat_zero_point": int(heat_d["quantization"][1]),
    "box_scale": float(box_d["quantization"][0]), "box_zero_point": int(box_d["quantization"][1]),
    "preprocess": "resize RGB -> x = pixel/127.5 - 1 -> q = round(x/input_scale + input_zero_point)",
    "box_layout": "box[y][x][4c..4c+3] = w, h, dx, dy (ô lưới); w = exp(w), h = exp(h) nếu wh_log",
    "wh_log": WH_LOG, "stride": INPUT_W // (INPUT_W // 8),
}, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------- 5. Dự đoán tập test bằng mô hình INT8 ----------------
test_s = load_samples(test_paths, None)
X_test = np.stack([preprocess(s["img"]) for s in test_s])
heat, box = tflite_predict(X_test)
preds = {s["id"]: decode(heat[i], box[i], s["orig_size"]) for i, s in enumerate(test_s)}
(OUT / "det_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
print(f"\nĐã lưu {OUT / 'det_predictions.json'} ({sum(map(len, preds.values()))} box, từ mô hình INT8)")
print("Tiếp theo: python step4_make_submission.py")
