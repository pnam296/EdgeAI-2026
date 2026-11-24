"""Ước tính thời gian Invoke() trên ESP32-S3 khi CHƯA có board, bằng cách đếm phép tính trong file .tflite và
so với model baseline (BTC đo trên board: detection 1389 ms).

CHỈ LÀ ƯỚC LƯỢNG: tốc độ thật phụ thuộc esp-nn có tối ưu op đó không, arena nằm PSRAM hay SRAM, cache...
Khi có board, số thật in trong log và trên web (ms_det). Dùng để so tương đối giữa các model.

    D:/edgeai/edgeai/Scripts/python.exe tools/estimate_latency.py                       # model trong main/models
    D:/edgeai/edgeai/Scripts/python.exe tools/estimate_latency.py a.tflite b.tflite
"""
from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tfl_utils import FW, MODELS, cc_to_bytes  # noqa: E402

BASELINE_MS = 1389.0      # README baseline: detect 1389 ms trên ESP32-S3 240 MHz

# Chi phí tương đối của 1 phần tử ra so với 1 MAC của conv (esp-nn S3). Giả định thô, sửa khi có số đo thật.
ELEM_COST = {
    "LOGISTIC": 12.0,      # int8 logistic của TFLM: tính xấp xỉ exp bằng số nguyên, không có esp-nn
    "MUL": 3.0, "ADD": 3.0, "QUANTIZE": 2.0, "CONCATENATION": 0.5, "PAD": 0.5, "SPLIT": 0.5,
    "RESIZE_NEAREST_NEIGHBOR": 1.0, "RESHAPE": 0.0, "SOFTMAX": 10.0,
}


def analyze(tfl: bytes):
    from tensorflow.lite.python import schema_py_generated as S
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(tfl, 0))
    names = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    g = m.subgraphs[0]
    shp = lambda i: [int(v) for v in g.tensors[i].shape]
    macs = defaultdict(float)
    elems = defaultdict(float)
    for op in g.operators:
        code = m.operatorCodes[op.opcodeIndex]
        name = names[max(code.builtinCode, code.deprecatedBuiltinCode)]
        out = np.prod(shp(op.outputs[0])) if len(op.outputs) else 0
        if name == "CONV_2D":
            kh, kw, cin = shp(op.inputs[1])[1:]
            macs[name] += out * kh * kw * cin
        elif name == "DEPTHWISE_CONV_2D":
            kh, kw = shp(op.inputs[1])[1:3]
            macs[name] += out * kh * kw
        elif name == "FULLY_CONNECTED":
            macs[name] += np.prod(shp(op.inputs[1]))
        elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            o = op.builtinOptions
            macs[name] += out * o.filterHeight * o.filterWidth * 0.25
        else:
            elems[name] += sum(np.prod(shp(i)) for i in op.outputs)
    return macs, elems


def cost(macs, elems):
    return sum(macs.values()) + sum(ELEM_COST.get(k, 1.0) * v for k, v in elems.items())


def main():
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        files = [MODELS / "det_model.cc", MODELS / "textcnn_model.cc"]
    # mốc baseline: det_model.cc của repo gốc (git HEAD) -> chi phí tương ứng 1389 ms
    try:
        base_cc = subprocess.run(["git", "-C", str(FW), "show", "b06f12e:main/models/det_model.cc"],
                                 capture_output=True, text=True, check=True).stdout
        tmp = FW / "build_host" / "baseline_det.cc"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text(base_cc)
        b_m, b_e = analyze(cc_to_bytes(tmp))
        per_unit = BASELINE_MS / cost(b_m, b_e)
        print(f"Mốc baseline: {sum(b_m.values()) / 1e6:.0f} M MAC = {BASELINE_MS:.0f} ms đo trên board "
              f"-> {per_unit * 1e6:.2f} ms / 1M đơn vị chi phí")
    except Exception as e:   # noqa: BLE001
        per_unit = BASELINE_MS / 146e6
        print(f"(không đọc được baseline từ git: {e}; dùng 146M MAC = 1389 ms)")

    for f in files:
        data = cc_to_bytes(f) if f.suffix == ".cc" else f.read_bytes()
        macs, elems = analyze(data)
        c = cost(macs, elems)
        print(f"\n{f.name}: {len(data):,} byte")
        for k, v in sorted(macs.items(), key=lambda t: -t[1]):
            print(f"  {k:<24} {v / 1e6:8.1f} M MAC")
        for k, v in sorted(elems.items(), key=lambda t: -t[1]):
            if v:
                print(f"  {k:<24} {v / 1e6:8.2f} M phần tử  (x{ELEM_COST.get(k, 1.0):g})")
        lo, hi = c * per_unit * 0.7, c * per_unit * 1.4
        print(f"  => ƯỚC TÍNH Invoke trên ESP32-S3: ~{c * per_unit:.0f} ms (khoảng {lo:.0f}..{hi:.0f} ms)")


if __name__ == "__main__":
    main()
