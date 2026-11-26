"""Chấm phần HỎI ĐÁP như trên ESP trên bộ đề có nhãn (vd data/intent_exam.csv: question,intent,classes).

Chuỗi xử lý giống firmware: bỏ dấu (như web) -> normalize (a-z0-9, còn lại thành dấu cách) -> tokenize theo
main/models/vocab.cc (id = vị trí + 2, từ lạ = 1, đệm 0) -> TextCNN INT8 -> ý định;
tên vật: gọi THẲNG hàm C logic_find_classes của main/logic.c (biên dịch thành DLL bằng gcc).

    python tools/eval_qa.py                       # model + vocab + KW đang nạp ESP
    python tools/eval_qa.py --exam ../data/intent_exam.csv --tflite x.tflite --vocab x.cc --show-ok
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tfl_utils import FW, MODELS  # noqa: E402

INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
CLASSES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", " ", strip_accents(s).lower())


def load_vocab(cc: Path):
    words = [l.strip().rstrip(",")[1:-1] for l in cc.read_text(encoding="utf-8").splitlines() if l.strip().startswith('"')]
    return {w: i + 2 for i, w in enumerate(words)}


def load_kw(logic_c: Path):
    src = logic_c.read_text(encoding="utf-8")
    body = src[src.index("KW[NUM_CLASSES][MAX_KW] = {"):]
    body = body[:body.index("};")]
    rows = re.findall(r"\{([^{}]*)\}", body)
    return [re.findall(r'"([^"]*)"', r) for r in rows]


def c_find_classes(logic_c: Path):
    """biên dịch main/logic.c thành DLL -> gọi đúng hàm tìm tên vật của firmware (khớp chính xác + gần đúng)"""
    import ctypes as C
    import subprocess
    out = FW / "build_host" / ("logic_qa.dll" if os.name == "nt" else "liblogic_qa.so")
    out.parent.mkdir(exist_ok=True)
    subprocess.check_call(["gcc", "-shared", "-O2", "-fPIC", "-static-libgcc", "-I", str(FW / "main"), str(logic_c),
                           "-lm", "-o", str(out)])
    lib = C.CDLL(str(out))
    lib.logic_find_classes.argtypes = [C.c_char_p, C.POINTER(C.c_int)]

    def f(norm: str):
        buf = (C.c_int * len(CLASSES))()
        n = lib.logic_find_classes(" ".join(norm.split()).encode(), buf)
        return [CLASSES[buf[i]] for i in range(n)]
    return f


def find_classes(norm: str, kw):
    words = " ".join(norm.split())
    found = []
    for c, keys in enumerate(kw):
        best = -1
        for k in keys:
            m = re.search(r"(?:^| )" + re.escape(k) + r"(?= |$)", words)
            if m and (best < 0 or m.start() < best):
                best = m.start()
        if best >= 0:
            found.append((best, CLASSES[c]))
    return [c for _, c in sorted(found)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", default=str(FW.parent / "data" / "intent_exam.csv"))
    ap.add_argument("--tflite", default=str(MODELS / "textcnn_model.tflite"))
    ap.add_argument("--vocab", default=str(MODELS / "vocab.cc"))
    ap.add_argument("--logic", default=str(FW / "main" / "logic.c"))
    ap.add_argument("--show-ok", action="store_true")
    a = ap.parse_args()

    import tensorflow as tf
    vocab = load_vocab(Path(a.vocab))
    find_c = c_find_classes(Path(a.logic))
    it = tf.lite.Interpreter(a.tflite)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    L = int(inp["shape"][1])

    rows = list(csv.DictReader(open(a.exam, encoding="utf-8")))
    res = defaultdict(Counter)
    errors = []
    oov = Counter()
    for r in rows:
        norm = normalize(r["question"])
        toks = norm.split()
        oov.update(t for t in toks if t not in vocab)
        ids = [vocab.get(t, 1) for t in toks][:L]
        ids += [0] * (L - len(ids))
        it.set_tensor(inp["index"], np.array([ids], dtype=inp["dtype"]))
        it.invoke()
        pred = INTENTS[int(np.argmax(it.get_tensor(out["index"])[0]))]
        want_cls = [c for c in r["classes"].split(";") if c]
        got_cls = find_c(norm)
        gi = r["intent"]
        ok_int = pred == gi
        needs_cls = gi in ("COUNT", "EXIST", "LOCATION")
        ok_cls = (not needs_cls) or got_cls == want_cls
        res[gi]["n"] += 1
        res[gi]["intent"] += ok_int
        res[gi]["classes"] += ok_cls
        res[gi]["both"] += ok_int and ok_cls
        if not (ok_int and ok_cls) or a.show_ok:
            why = []
            if not ok_int:
                why.append(f"ý định {gi} -> {pred}")
            if not ok_cls:
                why.append(f"vật {want_cls} -> {got_cls}")
            errors.append(f"  {'SAI' if why else 'ok '} {r['question']!r}: {'; '.join(why)}")

    tot = Counter()
    print(f"{'ý định':14s} {'số câu':>6s} {'ý định đúng':>11s} {'tên vật đúng':>12s} {'cả hai':>7s}")
    for gi in INTENTS:
        c = res[gi]
        if not c["n"]:
            continue
        tot.update(c)
        print(f"{gi:14s} {c['n']:6d} {c['intent'] / c['n']:11.2f} {c['classes'] / c['n']:12.2f} {c['both'] / c['n']:7.2f}")
    print(f"{'TỔNG':14s} {tot['n']:6d} {tot['intent'] / tot['n']:11.2f} {tot['classes'] / tot['n']:12.2f} {tot['both'] / tot['n']:7.2f}")
    print(f"\ntừ lạ (không có trong vocab) hay gặp: {', '.join(f'{w}({n})' for w, n in oov.most_common(15)) or '-'}")
    print(f"\n{len([e for e in errors if 'SAI' in e])} câu sai:")
    print("\n".join(errors))


if __name__ == "__main__":
    main()
