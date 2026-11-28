# Firmware chung kết: làm gì khi có ESP32-S3

Firmware này dựa trên repo baseline của BTC (`clbpioneer/edge-ai-challenge-2026-baseline`). Giữ nguyên sơ đồ chân,
WiFi, stream camera; thay model bằng model của đội và nối camera vào model.

| Phần | Baseline | Bản của đội |
|---|---|---|
| Detection | CNN grid 15×20, 1,4 s | YOLO P2P4 191k tham số (`yolo_kd_student.pt`), INT8 328 KB. Firmware tự nhận model grid hay YOLO |
| Intent | TextCNN 32 token | TextCNN của đội, 24 token, INT8 27 KB |
| Camera | chỉ stream, chưa nối vào model | `POST /run_cam`: chụp OV7670 → detect → intent → trả lời |
| Web | ảnh tải lên | nút **Run camera**, **tổng thời gian Detect + Intent**, bảng đối tượng + score, bộ 10 câu hỏi, lịch sử, tải CSV, chỉnh ngưỡng |
| Op TFLite Micro | viết tay | sinh tự động `main/models/ops_gen.h` từ model |

## Đã kiểm chứng trên PC (chưa có board)

- Keras dựng lại khớp PyTorch: sai số 2,6e-5.
- mAP@0.5 trên 1.200 ảnh val: float 0,9827 → INT8 0,9792 (NMS theo lớp, IoU 0,7). Cấu hình mặc định của firmware
  (NMS chung lớp, IoU 0,6) cho 0,9745. Hai bảng bên dưới dùng cấu hình mặc định này.
- Decode C thật của firmware (`main/logic.c`, gcc) cho mAP chỉ lệch decode Python 0,002, vì Python giữ được nhiều box
  điểm thấp hơn; số box conf ≥ 0,25: C 8.189, Python 8.191.
- Intent INT8 trùng float 6.000/6.000 câu, tách từ bằng C khớp Python 303/303 câu.
- Mọi op đều có trong TFLite Micro: không còn `TILE` (đổi upsample sang resize) hay `SELECT` (đổi embedding sang gather).
- `test/test_logic.c` pass. Phần C mới của `main.c`/`cam.c` qua kiểm tra cú pháp `-Wall -Wextra` với header giả lập.

**Chưa kiểm chứng được (cần board):** build bằng ESP-IDF, arena thực tế, **thời gian thật**, thứ tự byte RGB565
của camera, lệnh lật ảnh (thanh ghi MVFP).

Ước tính thời gian (`tools/estimate_latency.py`, lấy mốc baseline 1389 ms): detect khoảng **3,1 s** (dao động
2,2–4,4 s), intent khoảng 1 ms. Giới hạn là 5 s nên cần đo ngay khi có board. 57% phép tính nằm ở mức P2 (80×60).

## 0. Thử ngay hôm nay, không cần board

```bash
cd D:/edgeai/firmware
D:/edgeai/edgeai/Scripts/python.exe tools/mock_esp.py          # mở http://localhost:8080/
```
ESP giả lập dùng đúng trang web, đúng model INT8 và đúng code C hậu xử lý. "Camera" là thư mục ảnh val, mỗi lần chụp
chuyển sang ảnh kế. Dán 10 câu hỏi vào ô **Bộ câu hỏi**, tập bấm như lúc thi. `--webcam 0` để dùng webcam laptop.

## 1. Cài ESP-IDF (1 lần)

Windows: ESP-IDF Installation Manager (máy đã có `~/.espressif/eim_gui`), chọn **v5.5**, target esp32s3. Mở
"ESP-IDF PowerShell" (hoặc chạy `export.ps1`) để có `idf.py`.

## 2. Build + flash

```bash
cd D:/edgeai/firmware
idf.py set-target esp32s3
idf.py menuconfig        # ESP32-S3 Infer: WiFi (để trống = phát AP), lật ảnh camera, đổi byte RGB565
idf.py -p COM5 flash monitor
```
- Board PSRAM **Quad** (không phải N8R8/N16R8): trong `sdkconfig.defaults` đổi `CONFIG_SPIRAM_MODE_OCT=y` thành
  `CONFIG_SPIRAM_MODE_QUAD=y`, xóa `sdkconfig`, build lại.
- Thi chung kết: ESP phát AP, laptop nối WiFi `ESP32S3-INFER` (mật khẩu `12345678`), mở http://192.168.4.1/

Log khởi động đúng:
```
infer: detection OK | arena dung XXXX / 3072 KB
infer:   muc 0: 60x80 stride 4 (box=outA, cls=outB)
infer:   muc 1: 30x40 stride 8 ...
infer:   muc 2: 15x20 stride 16 ...
infer: detection head: yolo
infer: textcnn OK | arena dung ...
infer: vocab 261 tu, cau 24 token
```
Ghi lại số "arena dung" để chỉnh `DET_ARENA_KB` (`main/model_config.h`).

## 3. Thứ tự thử (khoảng 30 phút)

1. **Ảnh tải lên** (chưa cần camera): tải 1 ảnh val, chạy cùng câu hỏi trên web ESP và trên mock. Danh sách vật và
   câu trả lời phải gần như trùng, score lệch ±0,01. Nếu lệch nhiều thì model hoặc op có vấn đề.
2. **Thời gian**: xem ô *Detect + Intent* (xanh < 3 s, cam < 5 s, đỏ ≥ 5 s) và `invoke` trong dòng phụ.
   ```bash
   python tools/client.py bench cau_hoi.txt --repeat 3 --image anh.png     # trung bình / max, file CSV trong runs/
   ```
3. **Camera**: bấm **Run camera**. Khung kết quả vẽ **đúng khung model đã thấy** (lấy từ `/frame.rgb`).
   - Sai màu (đỏ ↔ xanh, hoặc nhiễu sọc): `menuconfig` → `CAM_SWAP_BYTES = 1`.
   - Ngược hoặc lộn ảnh: `CAM_HMIRROR` / `CAM_VFLIP`.
4. **Ngưỡng**: mở mục *Ngưỡng* trên web, đổi `conf` ngay không cần nạp lại firmware (API: `?conf=0.45`). Mặc định 0,5.

## 4. Nếu detect quá chậm (> khoảng 4 s)

Thử theo thứ tự, mỗi bước export lại rồi `verify_det.py` + đo lại:
1. Kiểm tra log: arena phải nằm trong PSRAM và CPU 240 MHz (`sdkconfig.defaults` đã đặt sẵn).
2. Model nhẹ hơn ở mức P2: giảm kênh hoặc bỏ 1 khối C3k2 ở P2 trong `yolo_cfg/student_p2p4.yaml` rồi train lại.
   Vòng chung kết được dùng pretrained/dữ liệu ngoài, nên có thể distill từ thầy hiện có.
3. Đổi SiLU sang ReLU: bỏ được LOGISTIC + MUL (khoảng 20% thời gian), cần train lại.

## 5. Ảnh camera thật (quan trọng nhất cho độ chính xác)

Ảnh OV7670 ám màu, nhiễu, khác góc so với dataset. Vòng chung kết **được dùng dữ liệu ngoài**, nên:
```bash
python tools/client.py capture --out ../edgeai/data/ov7670 --interval 2      # ảnh PNG 320x240 đúng như model thấy
```
Bày 7 loại vật như BTC sẽ bày, đổi vị trí và ánh sáng, chụp vài trăm ảnh, gán nhãn YOLO. Sau đó:
- fine-tune detector trên ảnh camera + dữ liệu cũ, rồi export lại;
- calibrate INT8 bằng ảnh camera: `export_det_yolo.py --calib-extra ../edgeai/data/ov7670`;
- chọn `conf` trên ảnh camera có nhãn: `verify_det.py --images <thư mục ảnh camera>` (cần `labels/` song song
  `images/`), rồi đặt `CONF_THRESH` trong `main/model_config.h`.

## Thay model

```bash
D:/edgeai/edgeai/Scripts/python.exe tools/export_det_yolo.py --weights <best.pt>   # -> det_model.cc/.h + ops_gen.h
D:/edgeai/edgeai/Scripts/python.exe tools/export_intent.py                         # -> textcnn_model.cc/.h + vocab.cc/.h
D:/edgeai/edgeai/Scripts/python.exe tools/verify_det.py                            # mAP INT8 + decode C, bảng ngưỡng
D:/edgeai/edgeai/Scripts/python.exe tools/estimate_latency.py                      # ước tính ms trên ESP
idf.py build flash
```

## API

| | |
|---|---|
| `POST /run_cam?conf=&iou=&agn=` | body = câu hỏi (không dấu). Chụp camera → JSON |
| `POST /run?...` | body = 230400 byte RGB888 + câu hỏi |
| `GET /frame.rgb` | khung RGB888 model vừa chạy |
| `GET /snap.rgb` | chụp khung mới, không chạy model |
| `GET /info` | head, ngưỡng mặc định, có camera không, PSRAM trống |

JSON: `ms_det`, `ms_intent`, **`ms_total` = det + intent**, `ms_invoke`, `ms_cap`, `dets[{c, s, b:[x1,y1,x2,y2]}]`,
`intent`, `answer`.

## File đã đổi so với baseline

- `main/logic.c/.h`: `decode_yolo` (DFL, so ngưỡng ngay trên int8, NMS chung lớp tùy chọn), tách từ theo độ dài
  model, câu trả lời theo nhiều vật trong một câu và vị trí 3×3.
- `main/infer.cc/.h`: tự nhận head grid hoặc YOLO theo shape output, đọc độ dài câu từ model, đo thời gian từng bước.
- `main/cam.c/.h`: `cam_capture_rgb888` (mutex với stream, đổi RGB565 sang RGB888), tạm dừng stream khi model chạy,
  lật ảnh.
- `main/main.c`: `/run_cam`, `/frame.rgb`, `/snap.rgb`, `/info`, ngưỡng qua URL.
- `main/index.html`, `main/Kconfig.projbuild`, `main/model_config.h`, `main/models/*`.
- Mới: `tools/` (export, verify, estimate, mock, client), `test/yolo_host.c`, `test/tok_host.c`.
