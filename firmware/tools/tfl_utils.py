"""Tiện ích chung cho các script export: ghi mảng C, liệt kê op, sinh REGISTER_OPS."""
from __future__ import annotations

import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):   # console Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FW = Path(__file__).resolve().parents[1]          # thư mục firmware
MODELS = FW / "main" / "models"


def to_cc(data: bytes, name: str, out_dir: Path = MODELS) -> None:
    """Ghi <name>.cc/.h giống ô export của baseline.ipynb (alignas 16, có <name>_len)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ",\n".join("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) for i in range(0, len(data), 16))
    (out_dir / f"{name}.h").write_text(
        f"#pragma once\nextern const unsigned char {name}[];\nextern const unsigned int {name}_len;\n")
    (out_dir / f"{name}.cc").write_text(
        f'#include "{name}.h"\nalignas(16) const unsigned char {name}[] = {{\n{rows}\n}};\n'
        f"const unsigned int {name}_len = {len(data)};\n")
    print(f"  -> {out_dir / name}.cc/.h ({len(data):,} byte)")


def vocab_to_cc(words: list[str], out_dir: Path = MODELS) -> None:
    """vocab[k] có token id k+2 (0 = pad, 1 = từ lạ), giống baseline."""
    for w in words:
        assert re.fullmatch(r"[a-z0-9_]+", w), f"từ vocab lạ: {w!r}"
    body = ",\n".join(f'  "{w}"' for w in words)
    (out_dir / "vocab.h").write_text("#pragma once\nextern const char *const vocab[];\nextern const int vocab_len;\n")
    (out_dir / "vocab.cc").write_text(
        f'#include "vocab.h"\nconst char *const vocab[] = {{\n{body}\n}};\nconst int vocab_len = {len(words)};\n')
    print(f"  -> {out_dir / 'vocab'}.cc/.h ({len(words)} từ)")


def list_ops(tflite_bytes: bytes) -> list[str]:
    """Tên op (BuiltinOperator) dùng trong model, ví dụ ['CONV_2D', 'LOGISTIC', ...]."""
    from tensorflow.lite.python import schema_py_generated as S
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(tflite_bytes, 0))
    names = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    ops = set()
    for oc in m.operatorCodes:
        code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
        if oc.customCode:
            raise SystemExit(f"Model có custom op {oc.customCode!r}: TFLite Micro không chạy được")
        ops.add(names[code])
    return sorted(ops)


# BuiltinOperator -> hàm Add...() của MicroMutableOpResolver (esp-tflite-micro 1.x)
_ADD = {
    "ADD": "AddAdd", "AVERAGE_POOL_2D": "AddAveragePool2D", "CAST": "AddCast", "CONCATENATION": "AddConcatenation",
    "CONV_2D": "AddConv2D", "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D", "DEQUANTIZE": "AddDequantize",
    "EXPAND_DIMS": "AddExpandDims", "FULLY_CONNECTED": "AddFullyConnected", "GATHER": "AddGather",
    "HARD_SWISH": "AddHardSwish", "LESS": "AddLess", "LOGISTIC": "AddLogistic", "MAX_POOL_2D": "AddMaxPool2D",
    "MEAN": "AddMean", "MUL": "AddMul", "PAD": "AddPad", "PADV2": "AddPadV2", "QUANTIZE": "AddQuantize",
    "REDUCE_MAX": "AddReduceMax", "RELU": "AddRelu", "RELU6": "AddRelu6", "RESHAPE": "AddReshape",
    "RESIZE_NEAREST_NEIGHBOR": "AddResizeNearestNeighbor", "SOFTMAX": "AddSoftmax", "SPLIT": "AddSplit",
    "SPLIT_V": "AddSplitV", "SQUEEZE": "AddSqueeze", "STRIDED_SLICE": "AddStridedSlice", "SUB": "AddSub",
    "TANH": "AddTanh", "TRANSPOSE": "AddTranspose", "SHAPE": "AddShape", "PACK": "AddPack",
    "UNPACK": "AddUnpack", "LEAKY_RELU": "AddLeakyRelu", "MAXIMUM": "AddMaximum", "MINIMUM": "AddMinimum",
}


def cc_to_bytes(cc: Path) -> bytes:
    """Đọc ngược mảng byte từ file .cc (để biết op của model đang thực sự được build vào firmware)."""
    body = cc.read_text().split("{", 1)[1].split("}", 1)[0]
    return bytes(int(h, 16) for h in re.findall(r"0x([0-9a-fA-F]{2})", body))


def write_ops_header(names=("det_model", "textcnn_model"), out: Path = MODELS / "ops_gen.h") -> list[str]:
    """Sinh main/models/ops_gen.h: REGISTER_OPS = hợp các op của các model .cc đang nằm trong main/models/.
    Gọi lại sau mỗi lần thay model (các script export tự gọi)."""
    ops = sorted({op for n in names for op in list_ops(cc_to_bytes(MODELS / f"{n}.cc"))})
    missing = [o for o in ops if o not in _ADD]
    if missing:
        raise SystemExit(f"Chưa biết hàm Add cho op {missing}: thêm vào _ADD trong tools/tfl_utils.py")
    lines = " \\\n".join(f"    (r).{_ADD[o]}();" for o in ops)
    out.write_text(
        "#pragma once\n/* SINH TU DONG boi tools/tfl_utils.py (dung sua tay). Model: "
        + ", ".join(f"{n}.cc" for n in names) + " */\n"
        f"#define RESOLVER_SIZE {len(ops)}\n#define REGISTER_OPS(r) do {{ \\\n{lines} \\\n}} while (0)\n")
    print(f"  -> {out} ({len(ops)} op: {' '.join(ops)})")
    return ops
