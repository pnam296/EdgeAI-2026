"""Chấm 1 vòng thử nghiệm cam3: tải output Kaggle -> xuất INT8 -> chấm độ đúng CÂU TRẢ LỜI trên ảnh model CHƯA thấy.

    python tools/eval_round.py A D            # chỉ các bản đã xong
    python tools/eval_round.py A B C D --skip-download

Mỗi bản: tải yolo_cam3<V>_last.pt + results_cam.json + log (tài khoản theo ACCOUNTS), xuất TFLite INT8
(calibrate bằng ảnh BTC + data/ov7670, KHÔNG dùng ảnh test), rồi tune_answer --eval với cấu hình firmware
(conf 0.45, NMS 0.45 gộp lớp, loc 0.10, bottle/cup/mouse riêng) trên:
  - BTC mẫu : 9 ảnh mẫu thật của BTC (data/btc_samples) - giống chung kết nhất
  - real55  : 55 ảnh camera thật đã kiểm nhãn (outputs/cam2_data/cam_real/{train,test})
So với model đang nạp ESP (firmware/main/models/det_model.tflite).
Token tài khoản thứ 2 đọc từ biến môi trường KAGGLE_TOKEN_FRIEND (không ghi vào file).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / "Scripts" / "python.exe") if os.name == "nt" else sys.executable
KAGGLE = str(REPO / "Scripts" / "kaggle.exe") if os.name == "nt" else "kaggle"
ACCOUNTS = {"A": "phngnam296", "B": "phngnam296", "C": "dohuuminhtri", "D": "dohuuminhtri",
            "R2A": "dohuuminhtri", "R2B": "dohuuminhtri", "R2C": "phngnam296",
            "R3A": "dohuuminhtri", "R3B": "dohuuminhtri"}
TESTS = {"BTC mẫu": ["data/btc_samples"],
         "real55": ["outputs/cam2_data/real55"],   # = cam_real/train + cam_real/test gộp lại
         "khăn200": ["data/cam_new_test"],         # 200 ảnh nền khăn trắng, người đã sửa nhãn (giống chung kết)
         "khăn875": ["data/cam_new_train"],        # 875 ảnh khăn trắng còn lại, người đã sửa nhãn
         "khăn1075": ["outputs/khan1075"]}          # 200 + 875 gộp (đều đã sửa nhãn)
CFG = "0.45,0.45,1,0.10"        # = firmware: conf 0.45, NMS 0.45 gộp lớp, LOC_FLOOR 0.10
CFG_CLS = "bottle=0.35,cup=0.65,mouse=0.7"  # = firmware: ngưỡng riêng từng lớp
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def env_for(acc):
    e = dict(os.environ, PYTHONIOENCODING="utf-8")
    e.pop("KAGGLE_API_TOKEN", None)
    if acc != "phngnam296":
        tok = os.environ.get("KAGGLE_TOKEN_FRIEND")
        assert tok, "đặt biến môi trường KAGGLE_TOKEN_FRIEND cho tài khoản " + acc
        e["KAGGLE_API_TOKEN"] = tok
    return e


def sh(cmd, env=None, quiet=True):
    r = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env or dict(os.environ, PYTHONIOENCODING="utf-8"),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode and not quiet:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def answer_acc(tflite, dist, sets):
    args = [PY, "firmware/tools/tune_answer.py", "--tflite", tflite, "--dist-scale", dist, "--btc", "", "--rf", "",
            "--eval", CFG, "--eval-cls", CFG_CLS, "--absent-weight", "0.1", "--out", "outputs/eval_round_tmp.json"]
    for d in sets:
        args += ["--yolo", d]
    out = sh(args).stdout
    m = re.search(r"tổng ([0-9.]+) \| C\+E ([0-9.]+).*?\| LIST ([0-9.]+) \| LOC ([0-9.]+)", out)
    return tuple(map(float, m.groups())) if m else None   # (tổng, count+exist, list, location)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()

    rows = []
    for v in a.variants:
        acc = ACCOUNTS[v]
        out = REPO / "kaggle_runs" / f"cam3{v}" / "out"
        if not a.skip_download:
            out.mkdir(parents=True, exist_ok=True)
            sh([KAGGLE, "kernels", "output", f"{acc}/edgeai-cam3{v.lower()}", "-p", out,
                    "--file-pattern", r"(yolo_cam3.*\.pt|results_cam\.json|\.log)$"], env=env_for(acc), quiet=False)
        res_f = out / "results_cam.json"
        if not res_f.exists():
            print(f"[{v}] chưa có results_cam.json (chưa xong / lỗi)")
            continue
        res = json.loads(res_f.read_text())
        cfg = res.get("cfg", {})
        dist = "1.5"
        w = out / f"yolo_cam3{v}_last.pt"
        tfl = REPO / "outputs" / f"cam3{v}" / "det_model.tflite"
        if not a.skip_export and w.exists():
            tfl.parent.mkdir(parents=True, exist_ok=True)
            sh([PY, "firmware/tools/export_det_yolo.py", "--weights", w, "--calib", "data/train/images",
                    "--calib-extra", "data/ov7670", "--out", tfl.parent, "--name", "det_model"], quiet=False)
            print(f"[{v}] xuất INT8: {'ok' if tfl.exists() else 'LỖI'}")
        accs = {k: answer_acc(tfl, dist, s) if tfl.exists() else None for k, s in TESTS.items()}
        st = res.get("student_last.pt", {})
        rows.append((f"cam3{v}", cfg, st, accs))

    # mốc: model đang nạp ESP
    base = {k: answer_acc(REPO / "firmware/main/models/det_model.tflite", "1.5", s) for k, s in TESTS.items()}
    print("\n" + "=" * 110)
    print(f"{'bản':8s} {'cấu hình':38s} | mAP50: {'BTCval':>6s} {'RF':>5s} {'real':>5s} {'BTCmẫu':>6s} | "
          f"đúng câu trả lời: {'BTC mẫu':>8s} {'real55':>7s} {'khăn200':>8s} {'khăn875':>8s} {'khăn1075':>9s}")
    fmt = lambda x: f"{x[0]:.3f}" if x else "  -  "
    for name, cfg, st, accs in rows:
        c = f"thay {cfg.get('teacher')} tro {cfg.get('s_yaml')}/{cfg.get('s_init') or 'tu dau'} {cfg.get('s_epochs')}ep"
        m = lambda s: f"{st[s]['map50']:.3f}" if s in st else "  -  "
        print(f"{name:8s} {c:38s} | {'':6s} {m('btc'):>6s} {m('rf'):>5s} {m('real'):>5s} {m('btcmau'):>6s} | "
              f"{'':17s} {fmt(accs['BTC mẫu']):>8s} {fmt(accs['real55']):>7s} {fmt(accs['khăn200']):>8s} {fmt(accs['khăn875']):>8s} {fmt(accs['khăn1075']):>9s}")
    print(f"{'ESP':8s} {'đang nạp firmware (main/models)':38s} | {'':34s} | {'':17s} "
          f"{fmt(base['BTC mẫu']):>8s} {fmt(base['real55']):>7s} {fmt(base['khăn200']):>8s} {fmt(base['khăn875']):>8s} {fmt(base['khăn1075']):>9s}")
    for name, cfg, st, accs in rows:
        if st.get("btcmau"):
            print(f"  {name} BTC mẫu theo lớp: {st['btcmau']['per_class']}")
            print(f"  {name} real    theo lớp: {st['real']['per_class']}")


if __name__ == "__main__":
    main()
