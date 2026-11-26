"""THỬ NGHIỆM TextCNN dùng embedding MẨU KÝ TỰ (kiểu fastText) - chống gõ sai / từ chưa gặp, <= 200k tham số.

Mỗi từ = mã băm cả từ + mã băm các mẩu 3-4 ký tự của "<từ>" (tối đa K mẩu), băm FNV-1a vào B ô (ô 0 = đệm, luôn 0).
Vector từ = TỔNG vector các ô -> từ gõ sai ('chaii', 'dien thaoi') vẫn chung phần lớn mẩu với từ đúng.
Bảng ô khởi tạo từ fastText 32d (pretrained): tối ưu để tổng vector các mẩu của mỗi từ fastText ~ vector gốc.
Dữ liệu train = y hệt tools/train_intent.py (v2); token 'oovunseen' -> chuỗi ký tự ngẫu nhiên.
Chưa đưa lên ESP: cần thêm hàm băm mẩu ký tự bằng C (logic.c) + op SUM.

    python tools/train_intent_ngram.py --out outputs/intent_ngram
"""
from __future__ import annotations

import argparse
import random
import re
import string
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train_intent as T  # noqa: E402

MAX_LEN, K, B, EMB = 24, 10, 5600, 32


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for ch in s.encode():
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return h


def word_feats(w: str):
    f = [1 + fnv1a("w:" + w) % (B - 1)]
    x = "<" + w + ">"
    for n in (3, 4):
        for i in range(len(x) - n + 1):
            if len(f) >= K:
                return f
            f.append(1 + fnv1a(x[i:i + n]) % (B - 1))
    return f


def encode(text: str):
    ids = np.zeros((MAX_LEN, K), np.int32)
    for i, w in enumerate(T.normalize(text).split()[:MAX_LEN]):
        f = word_feats(w)
        ids[i, :len(f)] = f
    return ids


def zero_pad_constraint():
    """ràng buộc: ô 0 (đệm) luôn = 0 -> tổng mẩu không bị ảnh hưởng. Đăng ký với Keras để load lại được model."""
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="edgeai")
    class ZeroPad(tf.keras.constraints.Constraint):
        def __call__(self, w):
            return tf.concat([tf.zeros_like(w[:1]), w[1:]], 0)
    return ZeroPad


def strip_acc(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(T.REPO / "outputs" / "intent_ngram"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    L = tf.keras.layers
    rng = random.Random(a.seed)

    # 1) khởi tạo bảng mẩu ký tự từ fastText 32d: min sum_w || sum_f E[f] - v_w ||^2
    ft = np.load(T.REPO / "pipeline" / "weights" / "vi_fasttext_32d.npz")
    words, vec = ft["words"].tolist(), ft["emb"].astype(np.float32)
    F = np.stack([np.pad(word_feats(w), (0, K - len(word_feats(w)))) for w in words])
    E = tf.Variable(np.random.default_rng(a.seed).normal(0, 0.05, (B, EMB)).astype(np.float32))
    opt = tf.keras.optimizers.Adam(0.05)
    Ft, Vt = tf.constant(F), tf.constant(vec)
    mask = tf.cast(Ft > 0, tf.float32)[..., None]
    for it in range(400):
        with tf.GradientTape() as g:
            pred = tf.reduce_sum(tf.gather(E, Ft) * mask, 1)
            loss = tf.reduce_mean(tf.reduce_sum((pred - Vt) ** 2, 1))
        opt.apply_gradients([(g.gradient(loss, E), E)])
    init = E.numpy()
    init[0] = 0
    print(f"khởi tạo mẩu ký tự từ {len(words)} từ fastText: sai số {float(loss):.4f} "
          f"(vector gốc TB {float(tf.reduce_mean(tf.reduce_sum(Vt ** 2, 1))):.4f})", flush=True)

    # 2) dữ liệu như v2
    df = pd.read_csv(T.REPO / "data" / "intent_train.csv")
    Tx, Ty = T.build_data(df, a.seed)
    gib = lambda: "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 7)))
    Tx = [re.sub(r"oovunseen", lambda m: gib(), t) for t in Tx]
    X = np.stack([encode(t) for t in Tx])
    y = np.array([T.INTENTS.index(v) for v in Ty], np.int32)

    ZeroPad = zero_pad_constraint()
    tin = L.Input((MAX_LEN, K), dtype="int32", name="tokens")
    e = L.Embedding(B, EMB, weights=[init], embeddings_constraint=ZeroPad())(tin)   # (24, K, 32)
    e = L.Lambda(lambda t: tf.reduce_sum(t, axis=2), name="sum_ngrams")(e)          # (24, 32)
    br = [L.GlobalMaxPooling1D()(L.Conv1D(32, k, activation="relu", padding="same")(e)) for k in (2, 3, 4)]
    h = L.Dropout(0.3)(L.Concatenate()(br))
    m = tf.keras.Model(tin, L.Dense(len(T.INTENTS), activation="softmax")(h))
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnt = np.bincount(y, minlength=5)
    m.fit(X, y, epochs=a.epochs, batch_size=64, class_weight={i: len(y) / (5 * c) for i, c in enumerate(cnt)},
          verbose=2, shuffle=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "intent_ngram.keras")
    print(f"TextCNN mẩu ký tự: {m.count_params():,} tham số -> {out}", flush=True)
    compare(m, a.seed)


def compare(ng, seed):
    """so với TextCNN v2 (theo từ, outputs/intent_v2) trên bộ đề sạch + bộ đề GÕ SAI ngẫu nhiên"""
    import tensorflow as tf
    ex = pd.read_csv(T.REPO / "data" / "intent_exam.csv")
    qs = [strip_acc(q) for q in ex.question]
    y = ex.intent.map(T.INTENTS.index).values
    v2 = tf.keras.models.load_model(T.REPO / "outputs" / "intent_v2" / "intent_textcnn.keras")
    itos = (T.REPO / "outputs" / "intent_v2" / "intent_vocab.txt").read_text(encoding="utf-8").splitlines()
    stoi = {w: i for i, w in enumerate(itos)}
    enc2 = lambda t: ([stoi.get(w, 1) for w in T.normalize(t).split()][:MAX_LEN] + [0] * MAX_LEN)[:MAX_LEN]

    def acc(model, encf, texts):
        p = model.predict(np.array([encf(t) for t in texts]), verbose=0).argmax(1)
        return (p == y).mean(), p

    def typo_set(sd, p):
        rng = random.Random(sd)
        return [" ".join(T.typo(w, rng) if len(w) >= 3 and rng.random() < p else w for w in T.normalize(q).split())
                for q in qs]
    print(f"{'bộ đề':24s} {'v2 (theo từ)':>12s} {'mẩu ký tự':>10s}")
    for name, texts in [("sạch (163 câu)", qs)] + [(f"gõ sai 30% (seed {k})", typo_set(k, 0.3)) for k in range(3)]             + [("gõ sai 60% (rất nặng)", typo_set(7, 0.6))]:
        a2, p2 = acc(v2, enc2, texts)
        an, pn = acc(ng, encode, texts)
        print(f"{name:24s} {a2:12.3f} {an:10.3f}", flush=True)
        if name.endswith("seed 0)"):
            for i in np.where((p2 != y) | (pn != y))[0][:10]:
                print(f"     {texts[i]!r}: đúng {T.INTENTS[y[i]]} | v2 {T.INTENTS[p2[i]]} | mẩu {T.INTENTS[pn[i]]}")


if __name__ == "__main__":
    main()
