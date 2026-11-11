"""BƯỚC 2 - Phân loại ý định câu hỏi bằng TextCNN.

Chạy:  python step2_intent_textcnn.py
Kết quả:
  outputs/intent_predictions.csv   -> dùng cho submission
  outputs/intent_textcnn.keras     -> mô hình Keras
  outputs/intent_int8.tflite       -> mô hình INT8 cho ESP32 (vòng chung kết)
  outputs/intent_vocab.txt         -> từ điển, cần mang theo lên thiết bị

KIẾN TRÚC TextCNN (Kim, 2014):
  token ids -> Embedding (mỗi từ thành vector) -> nhiều Conv1D song song với kích thước
  cửa sổ 2, 3, 4 từ (bắt cụm "bao nhieu", "co ... khong", "ben trai o giua")
  -> GlobalMaxPool (giữ tín hiệu mạnh nhất ở bất kỳ vị trí nào) -> Dense -> 5 nhãn.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from common import DATA, INTENTS, OUT
from intent_text import Vocab, normalize

SEED = 42
# ---- THỂ LỆ VÒNG LOẠI ----
# Chỉ dùng 63 câu train gốc; không từ khóa, không từ điển vật thể, không sinh thêm câu.
# Câu chỉ qua normalize (chữ thường, bỏ dấu câu, gộp khoảng trắng) rồi vào TextCNN.

MAX_LEN = 24      # vòng loại: câu dài nhất 21 từ (train + test)
EMB_DIM = 32
FILTERS = 32
KERNELS = (2, 3, 4)
# Vòng loại có 6.000 câu (Warm-up chỉ 63): 6000/64 ≈ 94 bước mỗi epoch -> 20 epoch là đủ.
# (Warm-up dùng EPOCHS = 150, BATCH = 8.)
EPOCHS = 20
BATCH = 64


def clean(text):
    """Chuẩn hóa dùng chung cho train / val / test / thiết bị."""
    return normalize(text)

tf.keras.utils.set_random_seed(SEED)
label2id = {l: i for i, l in enumerate(INTENTS)}


def build_model(vocab_size):
    inp = tf.keras.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")
    x = tf.keras.layers.Embedding(vocab_size, EMB_DIM)(inp)
    branches = []
    for k in KERNELS:
        b = tf.keras.layers.Conv1D(FILTERS, k, activation="relu", padding="same")(x)
        branches.append(tf.keras.layers.GlobalMaxPooling1D()(b))
    x = tf.keras.layers.Concatenate()(branches)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(INTENTS), activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def prepare(texts, labels, vocab=None):
    """Chuẩn hóa -> (xây vocab nếu chưa có) -> mã hóa thành mảng số."""
    aug_t, aug_l = [clean(t) for t in texts], list(labels)
    vocab = vocab or Vocab(aug_t)
    X = np.array([vocab.encode(t, MAX_LEN) for t in aug_t], dtype=np.int32)
    y = np.array([label2id[l] for l in aug_l], dtype=np.int32)
    present = np.unique(y)
    cw = compute_class_weight("balanced", classes=present, y=y)  # bù cho LIST / OUT_OF_SCOPE ít mẫu
    return X, y, vocab, dict(zip(present.tolist(), cw.tolist()))


def fit(X, y, cw, vocab_size):
    model = build_model(vocab_size)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, class_weight=cw, verbose=0, shuffle=True)
    return model


train = pd.read_csv(DATA / "intent_train.csv")
test = pd.read_csv(DATA / "intent_test.csv")
print("Phân bố nhãn train:", train.intent.value_counts().to_dict())

# ---------- 1. Đánh giá bằng 5-fold cross-validation ----------
# QUAN TRỌNG: vocab được xây lại ở MỖI fold, chỉ từ phần train của fold đó.
# Nếu xây trên cả 63 câu rồi mới chia, từ của câu val đã "lọt" vào -> CV ảo cao (data leakage).
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# LIST/OUT_OF_SCOPE quá ít mẫu cho 5 fold -> stratify theo nhãn nhưng fold có thể thiếu lớp, chấp nhận được
accs, wrong = [], []
for fold, (tr_idx, va_idx) in enumerate(skf.split(train.question, train.intent)):
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    X, y, vocab, cw = prepare(tr.question, tr.intent)
    model = fit(X, y, cw, len(vocab))
    Xv = np.array([vocab.encode(clean(q), MAX_LEN) for q in va.question], dtype=np.int32)
    pred = model.predict(Xv, verbose=0).argmax(1)
    yv = va.intent.map(label2id).values
    accs.append((pred == yv).mean())
    wrong += [(q, t, INTENTS[p]) for q, t, p, ok in zip(va.question, va.intent, pred, pred == yv) if not ok]
    print(f"  fold {fold + 1}: acc = {accs[-1]:.3f}")
print(f"CV accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Tổng số câu sai khi CV: {len(wrong)} / {len(train)}" + (" (in tối đa 30)" if len(wrong) > 30 else ""))
for q, t, p in wrong[:30]:
    print(f"  SAI: [{t} -> {p}] {q}")

# ---------- 2. Huấn luyện mô hình cuối trên toàn bộ train ----------
X, y, vocab, cw = prepare(train.question, train.intent)
model = fit(X, y, cw, len(vocab))
print(f"\nSố tham số: {model.count_params():,}")
model.save(OUT / "intent_textcnn.keras")
(OUT / "intent_vocab.txt").write_text("\n".join(vocab.itos), encoding="utf-8")

Xt = np.array([vocab.encode(clean(q), MAX_LEN) for q in test.question], dtype=np.int32)
print(f"\n{len(X)} câu train | vocab {len(vocab)} từ (tự xây từ câu train, không từ điển ngoài)")
print(f"Ví dụ chuẩn hóa: '{test.question[1]}' -> '{clean(test.question[1])}'")
probs = model.predict(Xt, verbose=0)
test["intent"] = [INTENTS[i] for i in probs.argmax(1)]
test["confidence"] = probs.max(1).round(3)
test[["question_id", "intent"]].to_csv(OUT / "intent_predictions.csv", index=False)
print("\nDự đoán tập test (kiểm tra bằng mắt - đây là học, KHÔNG sửa tay kết quả nộp):")
print("  Phân bố nhãn dự đoán:", test.intent.value_counts().to_dict(), "(dữ liệu test cân bằng: mỗi nhãn 800 câu)")
low = test[test.confidence < 0.8]
print(f"  Số câu độ tin cậy < 0.8: {len(low)}" + (" (in tối đa 30)" if len(low) > 30 else ""))
for _, r in low.head(30).iterrows():
    print(f"  {r.question_id} {r.intent:<13}{r.confidence:.2f}  {r.question}")
test[["question_id", "question", "intent", "confidence"]].to_csv(OUT / "intent_test_debug.csv", index=False)
print(f"  Toàn bộ dự đoán kèm độ tin cậy: {OUT / 'intent_test_debug.csv'}")

# ---------- 3. Xuất TFLite INT8 (chuẩn bị cho ESP32) ----------
def representative_data():
    for i in range(0, len(X), 4):
        yield [X[i:i + 1]]


try:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_output_type = tf.int8  # đầu vào giữ int32 (token id)
    tfl = conv.convert()
    path = OUT / "intent_int8.tflite"
    path.write_bytes(tfl)

    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
    agree = 0
    for row, ref in zip(Xt, probs.argmax(1)):
        interp.set_tensor(i_det["index"], row[None].astype(i_det["dtype"]))
        interp.invoke()
        agree += interp.get_tensor(o_det["index"])[0].argmax() == ref
    print(f"\nTFLite INT8: {len(tfl) / 1024:.1f} KB (giới hạn 1,5 MB) | "
          f"khớp với Keras {agree}/{len(Xt)} câu test")
except Exception as e:  # không ảnh hưởng submission
    print(f"\n[Bỏ qua xuất TFLite] {type(e).__name__}: {e}")

print(f"\nĐã lưu {OUT / 'intent_predictions.csv'}")
