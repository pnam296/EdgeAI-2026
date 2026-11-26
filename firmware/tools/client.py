"""Client cho ESP (hoặc tools/mock_esp.py): chạy bộ câu hỏi, đo thời gian, chụp ảnh camera làm dữ liệu.

    # 1 câu, dùng camera của ESP; lưu ảnh có box vào runs/
    python tools/client.py run "Co bao nhieu coc?" --host 192.168.4.1
    # 1 câu trên 1 ảnh từ máy
    python tools/client.py run "Co bao nhieu coc?" --image anh.png
    # cả bộ câu hỏi (mỗi dòng 1 câu), lặp 3 lần để đo thời gian trung bình -> runs/bench_*.csv
    python tools/client.py bench cau_hoi.txt --repeat 3 --conf 0.5
    # thu ảnh OV7670 để fine-tune / calibrate: 1 ảnh mỗi 2 s (Enter = chụp ngay, Ctrl+C = dừng)
    python tools/client.py capture --out ../edgeai/data/ov7670 --interval 2
Chỉ cần requests, numpy, opencv-python (đã có trong venv dự án).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

W, H = 320, 240
COLORS = [(75, 25, 230), (75, 180, 60), (216, 99, 67), (49, 130, 245), (180, 30, 145), (184, 159, 14), (230, 50, 240)]
NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def strip_vn(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(c if 32 <= ord(c) < 127 else " " for c in s)[:500]


class Esp:
    def __init__(self, host, timeout=15):
        self.base = host if host.startswith("http") else f"http://{host}"
        self.timeout = timeout
        self.s = requests.Session()

    def info(self):
        return self.s.get(f"{self.base}/info", timeout=self.timeout).json()

    def _params(self, conf, iou, agn):
        return {k: v for k, v in (("conf", conf), ("iou", iou), ("agn", agn)) if v is not None}

    def run(self, q, image=None, conf=None, iou=None, agn=None):
        qb = strip_vn(q).encode()
        t0 = time.perf_counter()
        if image is None:
            r = self.s.post(f"{self.base}/run_cam", data=qb, params=self._params(conf, iou, agn), timeout=self.timeout)
        else:
            r = self.s.post(f"{self.base}/run", data=image.tobytes() + qb, params=self._params(conf, iou, agn),
                            timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        j["ms_roundtrip"] = round((time.perf_counter() - t0) * 1000)
        return j

    def frame(self, snap=False):
        r = self.s.get(f"{self.base}/{'snap' if snap else 'frame'}.rgb", timeout=self.timeout)
        r.raise_for_status()
        return np.frombuffer(r.content, np.uint8).reshape(H, W, 3)


def load_image(p):
    im = cv2.imread(str(p))
    if im is None:
        raise SystemExit(f"không đọc được ảnh {p}")
    return np.ascontiguousarray(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1])


def annotate(rgb, dets, scale=2):
    im = cv2.resize(rgb[:, :, ::-1], (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    for d in dets:
        c = COLORS[NAMES.index(d["c"]) % len(COLORS)]
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(d["b"], (W * scale, H * scale) * 2))
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        cv2.putText(im, f"{d['c']} {d['s']:.2f}", (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return im


def summary(j):
    objs = {}
    for d in j["dets"]:
        objs[d["c"]] = objs.get(d["c"], 0) + 1
    return (f"det+intent {j['ms_total']:>5} ms (invoke {j.get('ms_invoke', '?')}, chụp {j.get('ms_cap', 0)}) "
            f"| cả vòng {j['ms_roundtrip']:>5} ms | {j['intent'] or '-':<12} | {j['answer']} "
            f"| {', '.join(f'{v} {k}' for k, v in objs.items()) or 'không có vật'}")


def cmd_run(a, esp):
    img = load_image(a.image) if a.image else None
    j = esp.run(a.question, img, a.conf, a.iou, a.agn)
    print(summary(j))
    for d in j["dets"]:
        print(f"   {d['c']:<8} {d['s']:.3f}  {d['b']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb = img if img is not None else esp.frame()
    cv2.imwrite(str(out / f"run_{stamp}.png"), annotate(rgb, j["dets"]))
    cv2.imwrite(str(out / f"frame_{stamp}.png"), rgb[:, :, ::-1])
    (out / f"run_{stamp}.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out / f'run_{stamp}.png'}")


def cmd_bench(a, esp):
    qs = [l.strip() for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    img = load_image(a.image) if a.image else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, over = [], 0
    for rep in range(a.repeat):
        for i, q in enumerate(qs, 1):
            j = esp.run(q, img, a.conf, a.iou, a.agn)
            over += j["ms_total"] > 5000
            print(f"[{rep + 1}.{i:02d}] {q[:50]:<50} {summary(j)}")
            rows.append([rep + 1, i, q, j["intent"], j["answer"], len(j["dets"]),
                         " ".join(f"{d['c']}:{d['s']:.2f}" for d in j["dets"]),
                         j["ms_det"], j["ms_intent"], j["ms_total"], j.get("ms_cap", 0), j["ms_roundtrip"]])
    tot = np.array([r[9] for r in rows])
    rt = np.array([r[11] for r in rows])
    print(f"\n{len(rows)} lần | detect+intent: TB {tot.mean():.0f} ms, max {tot.max()} ms | cả vòng TB {rt.mean():.0f} ms"
          f" | quá 5 s: {over}")
    f = out / f"bench_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(f, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["lan", "cau", "question", "intent", "answer", "n_obj", "objects", "ms_det", "ms_intent",
                    "ms_total", "ms_cap", "ms_roundtrip"])
        w.writerows(rows)
    print(f"-> {f}")


def cmd_capture(a, esp):
    """Chụp liên tục /snap.rgb -> PNG 320x240 đúng như model thấy (để gán nhãn, fine-tune, calibrate INT8)."""
    import threading
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    now = threading.Event()
    threading.Thread(target=lambda: [now.set() for _ in iter(sys.stdin.readline, "")], daemon=True).start()
    n = 0
    print(f"Lưu vào {out} | Enter = chụp ngay | Ctrl+C = dừng")
    try:
        while a.n <= 0 or n < a.n:
            try:
                rgb = esp.frame(snap=True)
            except Exception as e:   # ESP bận/wifi chập chờn: thử lại, không dừng cả buổi chụp
                print(f"  lỗi lấy ảnh ({type(e).__name__}), thử lại sau 3 s")
                time.sleep(3)
                continue
            p = out / f"{a.prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]
            cv2.imwrite(str(p) + ".png", rgb[:, :, ::-1])
            n += 1
            print(f"  {n}: {p.name}.png  (độ sáng TB {rgb.mean():.0f})")
            now.wait(a.interval)
            now.clear()
    except KeyboardInterrupt:
        pass
    print(f"Đã lưu {n} ảnh.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.4.1", help="IP của ESP (mock: localhost:8080)")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--iou", type=float)
    ap.add_argument("--agn", type=int)
    ap.add_argument("--out", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("question")
    r.add_argument("--image")
    b = sub.add_parser("bench")
    b.add_argument("questions", help="file txt, mỗi dòng 1 câu")
    b.add_argument("--image")
    b.add_argument("--repeat", type=int, default=1)
    c = sub.add_parser("capture")
    c.add_argument("--n", type=int, default=0, help="0 = tới khi Ctrl+C")
    c.add_argument("--interval", type=float, default=2.0)
    c.add_argument("--prefix", default="ov7670")
    sub.add_parser("info")
    a = ap.parse_args()
    esp = Esp(a.host)
    if a.cmd == "info":
        print(json.dumps(esp.info(), ensure_ascii=False, indent=1))
    else:
        {"run": cmd_run, "bench": cmd_bench, "capture": cmd_capture}[a.cmd](a, esp)


if __name__ == "__main__":
    main()
