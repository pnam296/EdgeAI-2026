"""ESP GIẢ LẬP trên laptop: thử web UI, client và bộ câu hỏi khi CHƯA có board.

Giống firmware ở những chỗ quan trọng:
  - cùng trang main/index.html, cùng API (/run, /run_cam, /frame.rgb, /snap.rgb, /info, :<port+1>/stream);
  - cùng model INT8 trong main/models/*.tflite;
  - hậu xử lý = CODE C THẬT main/logic.c (biên dịch thành DLL, gọi bằng ctypes): decode_yolo, NMS,
    tách từ, câu trả lời -> câu trả lời giống hệt trên ESP.
Khác: thời gian đo là của PC (ESP chậm hơn nhiều, xem tools/estimate_latency.py); "camera" là thư mục ảnh
(mỗi lần chụp chuyển sang ảnh kế, giống BTC đặt vật mới) hoặc webcam laptop.

    D:/edgeai/edgeai/Scripts/python.exe tools/mock_esp.py                      # ảnh val làm camera
    D:/edgeai/edgeai/Scripts/python.exe tools/mock_esp.py --cam-dir anh_ov7670/  # ảnh camera thật đã chụp
    D:/edgeai/edgeai/Scripts/python.exe tools/mock_esp.py --webcam 0
rồi mở http://localhost:8080/
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tfl_utils import FW, MODELS  # noqa: E402

W, H, NC = 320, 240, 7
CLASS = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
INTENT = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]


# ------------------------------------------------------------------ main/logic.c qua ctypes
class QT(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_int8)), ("scale", C.c_float), ("zp", C.c_int),
                ("H", C.c_int), ("W", C.c_int), ("C", C.c_int)]


class DP(C.Structure):
    _fields_ = [("conf", C.c_float), ("iou", C.c_float), ("agnostic", C.c_int)]


class DET(C.Structure):
    _fields_ = [("cls", C.c_int), ("score", C.c_float), ("x1", C.c_float), ("y1", C.c_float),
                ("x2", C.c_float), ("y2", C.c_float)]


def build_logic():
    out = FW / "build_host" / ("logic.dll" if os.name == "nt" else "liblogic.so")
    out.parent.mkdir(exist_ok=True)
    vocab_c = out.parent / "vocab_mock.c"
    vocab_c.write_text((MODELS / "vocab.cc").read_text())
    subprocess.check_call(["gcc", "-shared", "-O2", "-fPIC", "-static-libgcc", "-I", str(FW / "main"),
                           "-I", str(MODELS), str(FW / "main/logic.c"), str(vocab_c), "-lm", "-o", str(out)])
    lib = C.CDLL(str(out))
    lib.decode_yolo.argtypes = [C.POINTER(QT), C.POINTER(QT), C.c_int, C.c_int, C.c_int, C.c_int,
                                C.POINTER(DP), C.POINTER(DET), C.c_int]
    lib.make_answer.argtypes = [C.c_int, C.c_char_p, C.POINTER(DET), C.c_int, C.c_char_p, C.c_size_t]
    lib.tokenize.argtypes = [C.c_char_p, C.c_void_p, C.c_int, C.POINTER(C.c_int32), C.c_int]
    return lib


class Engine:
    def __init__(self, conf, iou, agn):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        self.lib = build_logic()
        # vocab là mảng con trỏ (const char *const vocab[]): truyền địa chỉ phần tử đầu
        self.vocab_ptr = C.addressof(C.c_void_p.in_dll(self.lib, "vocab"))
        self.vocab_len = C.c_int.in_dll(self.lib, "vocab_len").value
        self.det = tf.lite.Interpreter(model_path=str(MODELS / "det_model.tflite"))
        self.det.allocate_tensors()
        self.txt = tf.lite.Interpreter(model_path=str(MODELS / "textcnn_model.tflite"))
        self.txt.allocate_tensors()
        self.din = self.det.get_input_details()[0]
        outs = self.det.get_output_details()
        self.reg_max = max(o["shape"][3] for o in outs) // 4
        levels = sorted({o["shape"][1] for o in outs}, reverse=True)
        by = {(o["shape"][1], o["shape"][3]): o for o in outs}
        self.order = [(by[(h, 4 * self.reg_max)], by[(h, NC)]) for h in levels]
        self.seq = int(self.txt.get_input_details()[0]["shape"][1])
        self.defaults = dict(conf=conf, iou=iou, agn=agn)
        self.lock = threading.Lock()

    def detect(self, rgb, p):
        s, z = self.din["quantization"]
        x = np.clip(np.round(rgb.astype(np.float32) / 255.0 / s + z), -128, 127).astype(np.int8)[None]
        t0 = time.perf_counter()
        self.det.set_tensor(self.din["index"], x)
        self.det.invoke()
        t1 = time.perf_counter()
        keep, box, cls = [], (QT * 4)(), (QT * 4)()
        for i, (b, c) in enumerate(self.order):
            for dst, o in ((box, b), (cls, c)):
                t = np.ascontiguousarray(self.det.get_tensor(o["index"])[0])
                keep.append(t)   # giữ mảng sống tới khi decode xong
                dst[i] = QT(t.ctypes.data_as(C.POINTER(C.c_int8)), o["quantization"][0], o["quantization"][1],
                            *t.shape)
        dets = (DET * 64)()
        n = self.lib.decode_yolo(box, cls, len(self.order), W, H, self.reg_max,
                                 C.byref(DP(p["conf"], p["iou"], p["agn"])), dets, 64)
        t2 = time.perf_counter()
        return dets, n, (t1 - t0) * 1000, (t2 - t1) * 1000

    def intent(self, norm_q):
        ids = (C.c_int32 * 64)()
        self.lib.tokenize(norm_q, self.vocab_ptr, self.vocab_len, ids, self.seq)
        i0, o0 = self.txt.get_input_details()[0], self.txt.get_output_details()[0]
        self.txt.set_tensor(i0["index"], np.array([ids[:self.seq]], np.int32))
        self.txt.invoke()
        return int(self.txt.get_tensor(o0["index"])[0].argmax())

    def run(self, rgb, q: bytes, p, src, ms_cap=0):
        with self.lock:
            buf = C.create_string_buffer(q[:512], 513)
            self.lib.normalize_text(buf)
            norm = buf.value
            t0 = time.perf_counter()
            dets, n, ms_inv, ms_dec = self.detect(rgb, p)
            t1 = time.perf_counter()
            intent, answer = -1, b""
            if norm.strip():
                intent = self.intent(norm)
                ab = C.create_string_buffer(384)
                self.lib.make_answer(intent, norm, dets, n, ab, 384)
                answer = ab.value
            t2 = time.perf_counter()
        ms_det, ms_int = round((t1 - t0) * 1000), round((t2 - t1) * 1000)
        return {"src": src, "ms_cap": ms_cap, "ms_det": ms_det, "ms_intent": ms_int, "ms_total": ms_det + ms_int,
                "ms_fill": 0, "ms_invoke": round(ms_inv), "ms_decode": round(ms_dec),
                "conf": p["conf"], "iou": p["iou"], "agn": p["agn"],
                "dets": [{"c": CLASS[d.cls], "s": round(d.score, 3),
                          "b": [round(d.x1, 4), round(d.y1, 4), round(d.x2, 4), round(d.y2, 4)]} for d in dets[:n]],
                "q": norm.decode(), "intent": INTENT[intent] if intent >= 0 else "",
                "answer": answer.decode("utf-8")}


# ------------------------------------------------------------------ "camera"
class Camera:
    def __init__(self, cam_dir=None, webcam=None):
        self.cap = cv2.VideoCapture(webcam) if webcam is not None else None
        self.files = sorted(Path(cam_dir).glob("*.png")) + sorted(Path(cam_dir).glob("*.jpg")) if cam_dir else []
        self.i = 0
        self.cur = self._load(0) if self.files else None

    def _load(self, i):
        im = cv2.imread(str(self.files[i % len(self.files)]))
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()

    def live(self):   # khung hiện tại (BGR->RGB), cho stream
        if self.cap is not None:
            ok, f = self.cap.read()
            if ok:
                self.cur = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        return self.cur

    def capture(self):   # chụp cho model: thư mục ảnh -> sang ảnh kế
        if self.cap is None and self.files:
            self.i += 1
            self.cur = self._load(self.i)
        return self.live()


# ------------------------------------------------------------------ HTTP
def make_handler(eng: Engine, cam: Camera, stream_port: int):
    state = {"frame": None}
    html = (FW / "main" / "index.html").read_bytes()

    class Hd(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            if "/stream" not in self.path:
                sys.stderr.write("  %s\n" % (fmt % a))

        def send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def params(self):
            p = dict(eng.defaults)
            for k, v in parse_qs(urlparse(self.path).query).items():
                if k in p and v[0]:
                    p[k] = float(v[0]) if k != "agn" else int(v[0])
            return p

        def body(self):
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send(200, html, "text/html; charset=utf-8")
            if path == "/info":
                d = {"det": "yolo", "cam": cam.cur is not None, "mock": True, "stream_port": stream_port,
                     **eng.defaults}
                return self.send(200, json.dumps(d).encode(), "application/json")
            if path == "/frame.rgb":
                if state["frame"] is None:
                    return self.send(404, b"chua chay lan nao", "text/plain")
                return self.send(200, state["frame"].tobytes(), "application/octet-stream")
            if path == "/snap.rgb":
                f = cam.capture()
                if f is None:
                    return self.send(500, b"Khong co camera", "text/plain")
                state["frame"] = f
                return self.send(200, f.tobytes(), "application/octet-stream")
            if path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=fb")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while True:
                        f = cam.live()
                        if f is not None:
                            jpg = cv2.imencode(".jpg", f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                            self.wfile.write(b"\r\n--fb\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                             % len(jpg) + jpg)
                        time.sleep(0.1)
                except (ConnectionError, OSError):
                    return
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            data = self.body()
            if path == "/run":
                if not (W * H * 3 <= len(data) <= W * H * 3 + 512):
                    return self.send(400, b"body = 230400 byte RGB + cau hoi", "text/plain")
                rgb = np.frombuffer(data[:W * H * 3], np.uint8).reshape(H, W, 3)
                state["frame"] = rgb
                r = eng.run(rgb, data[W * H * 3:], self.params(), "upload")
            elif path == "/run_cam":
                t0 = time.perf_counter()
                f = cam.capture()
                if f is None:
                    return self.send(500, b"Khong co camera", "text/plain")
                state["frame"] = f
                r = eng.run(f, data, self.params(), "camera", round((time.perf_counter() - t0) * 1000))
            else:
                return self.send(404, b"not found", "text/plain")
            print(f"[{r['src']}] {len(r['dets'])} box | det {r['ms_det']} ms (PC) | {r['intent']} | {r['answer']}")
            self.send(200, json.dumps(r, ensure_ascii=False).encode("utf-8"), "application/json")

    return Hd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cam-dir", default=str(FW.parent / "edgeai/outputs/yolo_tiny/ds_split/images/val"))
    ap.add_argument("--webcam", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--agn", type=int, default=1)
    args = ap.parse_args()

    eng = Engine(args.conf, args.iou, args.agn)
    cam = Camera(None if args.webcam is not None else args.cam_dir, args.webcam)
    hd = make_handler(eng, cam, args.port + 1)
    s1 = ThreadingHTTPServer(("0.0.0.0", args.port), hd)
    s2 = ThreadingHTTPServer(("0.0.0.0", args.port + 1), hd)
    threading.Thread(target=s2.serve_forever, daemon=True).start()
    print(f"ESP giả lập: http://localhost:{args.port}/  (stream cổng {args.port + 1}, Ctrl+C để dừng)")
    try:
        s1.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
