"""Công cụ chạy thí nghiệm detector trên Kaggle (theo HUONG_DAN_TU_DONG_THI_NGHIEM.md).

  python kaggle_runs/_tools.py prepare <mã>     -> tạo kaggle_runs/<mã>/ (script đã sửa + kernel-metadata.json)
  python kaggle_runs/_tools.py parse <mã> "<thay đổi>"  -> đọc log trong kaggle_runs/<mã>/output, ghi results.csv
  python kaggle_runs/_tools.py accept <mã>      -> đưa thay đổi của <mã> vào mốc (state.json)
  python kaggle_runs/_tools.py gpu_hours        -> tổng giờ GPU đã dùng theo results.csv
"""
import csv
import json
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent
SRC = RUNS.parent / "edgeai_kaggle.py"
STATE = RUNS / "state.json"
RESULTS = RUNS / "results.csv"
USER = "phngnam296"
COMP = "egde-ai-challenge-2026-lan-2-qualifiers"
CLASSES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]

ALWAYS = [("\nQUICK_TEST = True\n", "\nQUICK_TEST = False\n")]
BIG_INPUT = [
    ("assert INPUT_W % 16 == 0 and INPUT_H % 16 == 0 and max(INPUT_W, INPUT_H) <= 160",
     "assert INPUT_W % 16 == 0 and INPUT_H % 16 == 0 and INPUT_W <= 320 and INPUT_H <= 240"),
    ('(f"Đầu vào {in_shape[1]}x{in_shape[0]} (mỗi chiều <= 160)", max(in_shape) <= 160)',
     '(f"Đầu vào {in_shape[1]}x{in_shape[0]} (<= 320x240)", in_shape[1] <= 320 and in_shape[0] <= 240)'),
]
INPUT_LINE = "INPUT_W, INPUT_H = 160, 128"
EXPERIMENTS = {
    "1a": [(INPUT_LINE, "INPUT_W, INPUT_H = 224, 160")] + BIG_INPUT,
    "1b": [(INPUT_LINE, "INPUT_W, INPUT_H = 320, 240")] + BIG_INPUT,
    "2a": [("ALPHA = 0.25", "ALPHA = 0.35")],
    "2b": [("NECK_CH = 64", "NECK_CH = 96")],
    "3": [("EPOCHS = 100", "EPOCHS = 150")],
    "5": [("s = rng.uniform(0.55, 1.0)", "s = rng.uniform(0.4, 1.0)")],
    "6a": [("LR = 2e-3", "LR = 3e-3")],
    "6b": [("BATCH = 16", "BATCH = 32")],
    "8": [("0.54 * w / 6, 0.4), max(0.54 * h / 6", "0.4 * w / 6, 0.4), max(0.4 * h / 6")],
    "final": [("TRAIN_ON_ALL = False", "TRAIN_ON_ALL = True")],
}
# Chạy lại độ phân giải trên mốc mới (ALPHA 0.35) -> cùng thay đổi, thư mục riêng
EXPERIMENTS["1a-a35"] = EXPERIMENTS["1a"]
EXPERIMENTS["1b-a35"] = EXPERIMENTS["1b"]
EXPERIMENTS["3-a35"] = EXPERIMENTS["3"]   # 150 epoch trên mốc 320x240 + α 0.35
# Người dùng chạy tay 3-a35 với 160 epoch (không phải 150) -> mốc dùng 160
EXPERIMENTS["3-a35-160"] = [("EPOCHS = 100", "EPOCHS = 160")]
# Dừng sớm + giảm LR khi chững (thay cosine), trần 400 epoch, trên mốc 3-a35-160
# ALPHA 0.35 -> 0.45 (~185k tham số) trên mốc 3-a35-160
EXPERIMENTS["2c"] = [("ALPHA = 0.35", "ALPHA = 0.45")]
EXPERIMENTS["es"] = [("EPOCHS = 160", "EPOCHS = 400"), ("PATIENCE = 0 ", "PATIENCE = 40 ")]
FIELDS = ["ma", "thay_doi", "tham_so", "s_per_epoch", "epoch_tot_nhat", "map_float", "map_int8",
          *[f"ap_{c}" for c in CLASSES], "tflite_kb", "ops", "rang_buoc_ok", "gpu", "nhan", "submission_ok",
          "gio_gpu", "ghi_chu"]
CONFIG_RE = re.compile(r"^(INPUT_W|ALPHA|NECK_CH|WH_LOG|EPOCHS|BATCH|LR|TRAIN_ON_ALL|QUICK_TEST|DICT_AUGMENT)\b"
                       r"|s = rng.uniform\(0\.|sx, sy = max|^assert INPUT_W|Đầu vào \{in_shape")


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8-sig")) if STATE.exists() else {"accepted": []}


def apply(code, pairs):
    for old, new in pairs:
        n = code.count(old)
        if n != 1:
            raise SystemExit(f"LỖI: chuỗi {old!r} xuất hiện {n} lần (cần đúng 1)")
        code = code.replace(old, new)
    return code


def prepare(exp):
    state = load_state()
    code = SRC.read_text(encoding="utf-8")
    pairs = list(ALWAYS)
    for acc in state["accepted"]:
        pairs += EXPERIMENTS[acc]
    # thí nghiệm đổi INPUT khi mốc đã đổi INPUT: bỏ thay thế INPUT/BIG_INPUT cũ, dùng cái mới
    exp_pairs = EXPERIMENTS[exp]
    if any(o == INPUT_LINE for o, _ in exp_pairs):
        pairs = [p for p in pairs if p[0] != INPUT_LINE and p not in BIG_INPUT]
    for p in exp_pairs:
        if p not in pairs:
            pairs.append(p)
    code = apply(code, pairs)
    d = RUNS / exp
    d.mkdir(exist_ok=True)
    (d / "edgeai_kaggle.py").write_text(code, encoding="utf-8")
    slug = f"edgeai-exp-{exp}"
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": f"edgeai exp {exp}", "code_file": "edgeai_kaggle.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False, "competition_sources": [COMP], "dataset_sources": [], "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    print(f"== {d} | mốc: {state['accepted'] or '(gốc)'} + {exp}")
    for i, line in enumerate(code.splitlines(), 1):
        if CONFIG_RE.search(line.strip()):
            print(f"  {i:>4}: {line.strip()}")


def read_log(exp):
    """Log Kaggle (.log) là JSON list {stream_name, time, data}. Trả về (text, giây chạy)."""
    out = RUNS / exp / "output"
    logs = sorted(out.rglob("*.log"))
    if not logs:
        raise SystemExit(f"Không thấy file .log trong {out}")
    raw = logs[0].read_text(encoding="utf-8", errors="replace")
    try:
        items = json.loads(raw)
        return "".join(i.get("data", "") for i in items), max((i.get("time", 0) for i in items), default=0)
    except json.JSONDecodeError:
        return raw, 0


def parse(exp, change):
    text, secs = read_log(exp)
    g = lambda pat, default="": (m.group(1).strip() if (m := re.search(pat, text)) else default)
    epochs = re.findall(r"epoch\s+\d+:.*?\|\s*([\d.]+)s/epoch", text)
    ap = {}
    for c in CLASSES:
        m = re.search(rf"^\s*{c}\s+([\d.]+)\s+([\d.]+)", text, re.M)
        ap[c] = m.group(1) if m else ""
    checks = re.findall(r"\[(OK|LỖI)\]", text)
    row = {
        "ma": exp, "thay_doi": change,
        "tham_so": g(r"Số tham số: ([\d,]+)").replace(",", ""),
        "s_per_epoch": epochs[-1] if epochs else "",
        "epoch_tot_nhat": g(r"Tốt nhất: epoch (\d+)"),
        "map_float": g(r"mAP50 val: float32 ([\d.]+)") or g(r"mAP50 val \(float32\): ([\d.]+)"),
        "map_int8": g(r"-> INT8 ([\d.]+)"),
        **{f"ap_{c}": ap[c] for c in CLASSES},
        "tflite_kb": g(r"Dung lượng (\d+) KB"),
        "ops": g(r"Op TFLite Micro: ([^\n]+)"),
        "rang_buoc_ok": "OK" if len(checks) >= 3 and all(c == "OK" for c in checks) else f"LỖI {checks}",
        "gpu": g(r"\| GPU: ([^\n]+)"),
        "nhan": g(r"=> Định dạng nhãn: (\S+)"),
        "submission_ok": "OK" if "OK - file hợp lệ" in text else "LỖI",
        "gio_gpu": f"{secs / 3600:.2f}",
        "ghi_chu": (f"mAP TTA lật ngang {t}" if (t := g(r"mAP50 val TTA lật ngang: ([\d.]+)")) else ""),
    }
    write_row(row)
    for k, v in row.items():
        print(f"  {k:<16}{v}")
    problems = []
    if "KHÔNG CÓ" in row["gpu"] or not row["gpu"]:
        problems.append("KHÔNG CÓ GPU")
    if row["nhan"] != "yolo":
        problems.append(f"định dạng nhãn = {row['nhan']!r}")
    if row["rang_buoc_ok"] != "OK":
        problems.append("ràng buộc LỖI")
    if row["submission_ok"] != "OK":
        problems.append("submission LỖI")
    print("KIỂM TRA BẮT BUỘC:", "ĐẠT" if not problems else "KHÔNG ĐẠT -> " + "; ".join(problems))


def write_row(row):
    rows = [] if not RESULTS.exists() else [r for r in csv.DictReader(RESULTS.open(encoding="utf-8"))
                                            if r["ma"] != row["ma"]]
    rows.append(row)
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)


def manual(exp, change, *kv):
    """Ghi kết quả chạy tay: manual <mã> "<thay đổi>" key=value ..."""
    row = {"ma": exp, "thay_doi": change, **dict(x.split("=", 1) for x in kv)}
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise SystemExit(f"Cột lạ: {unknown}")
    write_row(row)
    print(row)


def accept(exp):
    s = load_state()
    if exp not in s["accepted"]:
        s["accepted"].append(exp)
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("Mốc hiện tại:", s["accepted"])


def gpu_hours():
    if not RESULTS.exists():
        print("0.00")
        return
    print(f"{sum(float(r['gio_gpu'] or 0) for r in csv.DictReader(RESULTS.open(encoding='utf-8'))):.2f}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"prepare": prepare, "parse": parse, "manual": manual, "accept": accept,
     "gpu_hours": gpu_hours}[cmd](*args)
