"""Export TextCNN intent của đội (Keras .keras + vocab .txt) -> TFLite INT8 -> main/models/textcnn_model.cc/.h + vocab.cc/.h

Model Keras đã có sẵn NHWC/int32 nên convert thẳng, không qua onnx2tf (README baseline giải thích vì sao onnx2tf hỏng).
Kiểm tra: (1) INT8 so với float trên toàn bộ intent_train.csv, (2) tách từ kiểu C (main/logic.c) cho ra đúng
token id như Python.

    D:/edgeai/edgeai/Scripts/python.exe tools/export_intent.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfl_utils import FW, MODELS, list_ops, to_cc, vocab_to_cc, write_ops_header  # noqa: E402

ROOT = FW.parent / "edgeai"
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]   # thứ tự = INTENT_NAMES trong logic.c


def normalize(text):   # = pipeline.ipynb
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "pipeline/weights/intent_textcnn.keras"))
    ap.add_argument("--vocab", default=str(ROOT / "pipeline/weights/intent_vocab.txt"))
    ap.add_argument("--data", default=str(ROOT / "data/intent_train.csv"))
    ap.add_argument("--out", default=str(MODELS))
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    itos = Path(args.vocab).read_text(encoding="utf-8").split("\n")
    itos = [w for w in itos if w != ""]
    assert itos[:2] == ["<pad>", "<oov>"], "vocab phải bắt đầu bằng <pad>, <oov>"
    stoi = {w: i for i, w in enumerate(itos)}
    km = tf.keras.models.load_model(args.model, compile=False)
    seq = int(km.input_shape[1])
    print(f"[1/4] {Path(args.model).name}: {km.count_params():,} tham số, câu {seq} token, vocab {len(itos)} "
          f"(gồm pad/oov), output {km.output_shape}")
    assert km.output_shape[-1] == len(INTENTS)

    def encode(t):
        ids = [stoi.get(w, 1) for w in normalize(t).split()][:seq]
        return ids + [0] * (seq - len(ids))

    df = pd.read_csv(args.data)
    X = np.array([encode(q) for q in df.question], np.int32)
    y = np.array([INTENTS.index(v) for v in df.intent])
    # Dựng lại với batch 1 cho TFLite Micro. Embedding của Keras 3 xử lý id âm bằng LESS/ADD/SELECT
    # (Micro không có SELECT) -> thay bằng tf.gather thuần (chỉ ra 1 op GATHER), dùng lại các lớp còn lại.
    L = tf.keras.layers
    layers = {type(l).__name__: [] for l in km.layers}
    for l in km.layers:
        layers[type(l).__name__].append(l)
    assert set(layers) <= {"InputLayer", "Embedding", "Conv1D", "GlobalMaxPooling1D", "Concatenate", "Dropout",
                           "Dense"}, f"kiến trúc lạ: {set(layers)}"
    W_emb = layers["Embedding"][0].get_weights()[0]
    inp = tf.keras.Input((seq,), batch_size=1, dtype="int32", name="tokens")
    class Gather(L.Layer):
        def build(self, _):
            self.w = self.add_weight(shape=W_emb.shape, initializer=tf.keras.initializers.Constant(W_emb),
                                     trainable=False, name="table")

        def call(self, t):
            return tf.gather(self.w, t)

    e = Gather(name="emb")(inp)
    br = [g(c(e)) for c, g in zip(layers["Conv1D"], layers["GlobalMaxPooling1D"])]
    h = layers["Concatenate"][0](br) if len(br) > 1 else br[0]
    km1 = tf.keras.Model(inp, layers["Dense"][-1](h))
    chk = X[:256]
    d = np.abs(km1.predict(chk, batch_size=1, verbose=0) - km.predict(chk, verbose=0)).max()
    assert d < 1e-5, f"model dựng lại lệch {d}"
    pf = km.predict(X, batch_size=512, verbose=0).argmax(1)
    print(f"[2/4] float: accuracy trên {len(df)} câu intent_train = {(pf == y).mean():.4f}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))[:500]
    conv = tf.lite.TFLiteConverter.from_keras_model(km1)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([X[i:i + 1]] for i in idx)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_output_type = tf.int8          # input giữ int32 (token id)
    tfl = conv.convert()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "textcnn_model.tflite").write_bytes(tfl)
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
    assert i0["dtype"] == np.int32 and list(i0["shape"]) == [1, seq], i0
    pq = []
    for i in range(len(X)):
        it.set_tensor(i0["index"], X[i:i + 1])
        it.invoke()
        pq.append(int(it.get_tensor(o0["index"])[0].argmax()))
    pq = np.array(pq)
    print(f"[3/4] INT8: {len(tfl):,} byte, op: {' '.join(list_ops(tfl))}")
    print(f"      accuracy {(pq == y).mean():.4f}, trùng float {(pq == pf).mean():.4f}")

    # Tách từ kiểu C phải ra đúng token id như Python
    build = FW / "build_host"
    build.mkdir(exist_ok=True)
    words = itos[2:]
    vocab_to_cc(words, build)
    sample = list(df.question.sample(300, random_state=0)) + ["Co bao nhieu COC??", "  ", "toy_car o dau"]
    (build / "tok_in.txt").write_text("\n".join(normalize(q) for q in sample) + "\n", encoding="utf-8")
    exe = build / ("tok_host.exe" if os.name == "nt" else "tok_host")
    subprocess.check_call(["gcc", "-O1", "-I", str(FW / "main"), "-I", str(build), "-x", "c",
                           str(FW / "test/tok_host.c"), str(FW / "main/logic.c"), str(build / "vocab.cc"),
                           "-x", "none", "-lm", "-o", str(exe)])
    got = subprocess.run([str(exe), str(seq), str(build / "tok_in.txt")], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    bad = [(q, g) for q, g in zip(sample, got) if [int(v) for v in g.split()] != encode(q)]
    print(f"      tách từ C vs Python: {len(sample) - len(bad)}/{len(sample)} câu khớp")
    if bad:
        print("      LỆCH:", bad[:3])
        raise SystemExit(1)

    print("[4/4] Ghi mảng C:")
    to_cc(tfl, "textcnn_model", out)
    vocab_to_cc(words, out)
    if out.resolve() == MODELS.resolve():
        write_ops_header()


if __name__ == "__main__":
    main()
