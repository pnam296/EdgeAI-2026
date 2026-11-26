"""Host CLI tương tác trên máy tính để test phần Intent + câu trả lời (không cần board ESP32-S3).

Giống firmware ở những chỗ quan trọng:
  - Dùng CODE C THẬT trong main/logic.c (biên dịch thành DLL, gọi qua ctypes):
    normalize_text(), tokenize(), make_answer();
  - Mặc định dùng model TFLite INT8 (main/models/textcnn_model.tflite) + vocab.cc giống hệt trên ESP32-S3;
  - Hỗ trợ tùy chọn --keras để test nhanh model float từ pipeline/weights/intent_textcnn.keras;
  - Tự động bỏ dấu tiếng Việt (giống web index.html) nên gõ có dấu hay không dấu đều được.

Cách chạy:
    .venv\\Scripts\\python.exe firmware/tools/intent_host.py
    .venv\\Scripts\\python.exe firmware/tools/intent_host.py --keras
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tfl_utils import FW, MODELS  # noqa: E402

INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
CLASSES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


class DET(C.Structure):
    _fields_ = [("cls", C.c_int), ("score", C.c_float), ("x1", C.c_float), ("y1", C.c_float),
                ("x2", C.c_float), ("y2", C.c_float)]


def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt giống trình duyệt ở main/index.html trước khi gửi xuống ESP."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def build_logic():
    out = FW / "build_host" / ("logic_intent.dll" if os.name == "nt" else "liblogic_intent.so")
    out.parent.mkdir(exist_ok=True)
    vocab_c = out.parent / "vocab_intent_host.c"
    vocab_c.write_text((MODELS / "vocab.cc").read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.check_call(
        ["gcc", "-shared", "-O2", "-fPIC", "-static-libgcc", "-I", str(FW / "main"),
         "-I", str(MODELS), str(FW / "main/logic.c"), str(vocab_c), "-lm", "-o", str(out)]
    )
    lib = C.CDLL(str(out))
    lib.normalize_text.argtypes = [C.c_char_p]
    lib.tokenize.argtypes = [C.c_char_p, C.c_void_p, C.c_int, C.POINTER(C.c_int32), C.c_int]
    lib.make_answer.argtypes = [C.c_int, C.c_char_p, C.POINTER(DET), C.c_int, C.c_void_p, C.c_char_p, C.c_size_t]   # ap = NULL: dung nguyen box
    return lib


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def make_default_dets():
    """Tạo bộ đối tượng giả lập trong ảnh (giống test_logic.c: 2 cup + 1 book) để test make_answer()."""
    sample = [
        DET(1, 0.95, 0.05, 0.40, 0.25, 0.70),   # cup bên trái
        DET(1, 0.90, 0.75, 0.40, 0.95, 0.70),   # cup bên phải
        DET(3, 0.85, 0.40, 0.10, 0.60, 0.30),   # book phía trên ở giữa
    ]
    arr = (DET * len(sample))(*sample)
    return arr, len(sample)


def parse_custom_dets(spec: str):
    """Phân tích chuỗi dạng 'cup,book,phone' thành danh sách DET mẫu."""
    items = [s.strip() for s in spec.split(",") if s.strip()]
    dets = []
    for i, name in enumerate(items):
        if name not in CLASSES:
            raise ValueError(f"Lớp '{name}' không hợp lệ. Chọn trong: {', '.join(CLASSES)}")
        cx = 0.2 + (i % 3) * 0.3
        cy = 0.2 + ((i // 3) % 3) * 0.3
        dets.append(DET(CLASSES.index(name), 0.95 - i * 0.01, cx - 0.1, cy - 0.1, cx + 0.1, cy + 0.1))
    arr = (DET * max(1, len(dets)))(*dets)
    return arr, len(dets)


class IntentHost:
    def __init__(self, use_keras: bool = False):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")

        self.lib = build_logic()
        self.vocab_ptr = C.addressof(C.c_void_p.in_dll(self.lib, "vocab"))
        self.vocab_len = C.c_int.in_dll(self.lib, "vocab_len").value
        vocab_arr = (C.c_char_p * self.vocab_len).from_address(self.vocab_ptr)
        self.vocab_words = [vocab_arr[i].decode("utf-8") for i in range(self.vocab_len)]

        self.use_keras = use_keras
        if use_keras:
            kpath = FW.parent / "pipeline" / "weights" / "intent_textcnn.keras"
            self.model = tf.keras.models.load_model(str(kpath), compile=False)
            self.seq = int(self.model.input_shape[1])
            self.mode_name = f"Keras float ({kpath.name})"
        else:
            tpath = MODELS / "textcnn_model.tflite"
            self.it = tf.lite.Interpreter(model_path=str(tpath))
            self.it.allocate_tensors()
            self.i0 = self.it.get_input_details()[0]
            self.o0 = self.it.get_output_details()[0]
            self.seq = int(self.i0["shape"][1])
            self.mode_name = f"TFLite INT8 ({tpath.name})"

    def predict(self, question: str, dets, n_dets: int):
        ascii_q = strip_accents(question).encode("ascii", errors="ignore")[:512]
        buf = C.create_string_buffer(ascii_q, 513)
        self.lib.normalize_text(buf)
        norm_q = buf.value
        norm_str = norm_q.decode("ascii")

        ids = (C.c_int32 * 64)()
        self.lib.tokenize(norm_q, self.vocab_ptr, self.vocab_len, ids, self.seq)
        token_ids = [int(ids[i]) for i in range(self.seq)]

        if not norm_str.strip():
            return norm_str, token_ids, None, -1, "(Câu hỏi rỗng sau khi chuẩn hóa)", 0.0

        x = np.array([token_ids], dtype=np.int32)
        t0 = time.perf_counter()
        if self.use_keras:
            probs = self.model.predict(x, verbose=0)[0]
        else:
            self.it.set_tensor(self.i0["index"], x)
            self.it.invoke()
            raw = self.it.get_tensor(self.o0["index"])[0]
            if self.o0["dtype"] == np.int8:
                scale, zp = self.o0["quantization"]
                deq = (raw.astype(np.float32) - zp) * scale
                probs = softmax(deq) if abs(deq.sum() - 1.0) > 0.05 else np.clip(deq, 0.0, 1.0)
            else:
                probs = raw.astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000

        intent_idx = int(np.argmax(probs))
        ans_buf = C.create_string_buffer(384)
        self.lib.make_answer(intent_idx, norm_q, dets, n_dets, None, ans_buf, 384)
        answer = ans_buf.value.decode("utf-8", errors="replace")
        return norm_str, token_ids, probs, intent_idx, answer, ms


def format_dets_summary(dets, n_dets: int) -> str:
    if n_dets == 0:
        return "(trống - không có vật thể)"
    counts = {}
    for i in range(n_dets):
        name = CLASSES[dets[i].cls]
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{v} {k}" for k, v in counts.items())


def main():
    ap = argparse.ArgumentParser(description="Host CLI test Intent + logic.c từ bàn phím")
    ap.add_argument("--keras", action="store_true", help="Dùng pipeline/weights/intent_textcnn.keras thay cho TFLite INT8")
    args = ap.parse_args()

    host = IntentHost(use_keras=args.keras)
    dets, n_dets = make_default_dets()

    print("=" * 68)
    print(f"  INTENT HOST CLI | Model: {host.mode_name}")
    print(f"  Vocab C: {host.vocab_len} từ | Độ dài câu: {host.seq} token")
    print(f"  Ảnh giả lập hiện tại: {format_dets_summary(dets, n_dets)}")
    print("  Lệnh nhanh: ':det cup,book' đổi vật thể | ':empty' ảnh trống | ':q' thoát")
    print("=" * 68)

    while True:
        try:
            q = input("\nNhập câu hỏi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not q:
            continue
        if q in (":q", ":quit", ":exit", "exit", "quit"):
            print("Thoát.")
            break
        if q == ":empty":
            dets, n_dets = (DET * 1)(), 0
            print(f"-> Đã đổi ảnh giả lập: {format_dets_summary(dets, n_dets)}")
            continue
        if q.startswith(":det "):
            try:
                dets, n_dets = parse_custom_dets(q[5:])
                print(f"-> Đã đổi ảnh giả lập: {format_dets_summary(dets, n_dets)}")
            except ValueError as e:
                print(f"Lỗi: {e}")
            continue

        norm_str, token_ids, probs, intent_idx, answer, ms = host.predict(q, dets, n_dets)
        words = norm_str.split()[:host.seq]
        tok_view = [f"{w}({'OOV:1' if tid == 1 else tid})" for w, tid in zip(words, token_ids)]

        print(f"  Chuẩn hóa : {norm_str}")
        print(f"  Tokenize  : {' '.join(tok_view) if tok_view else '(rỗng)'}")
        if probs is not None:
            scores = " | ".join(f"{name}:{p*100:5.1f}%" for name, p in zip(INTENTS, probs))
            print(f"  Điểm số   : {scores}")
            print(f"  Intent    : {INTENTS[intent_idx]} ({probs[intent_idx]*100:.1f}%) | {ms:.2f} ms")
        print(f"  Trả lời   : {answer}")


if __name__ == "__main__":
    main()
