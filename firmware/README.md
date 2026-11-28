# Firmware thi đấu – ESP32-S3 + OV7670

Chụp ảnh từ camera → phát hiện đồ vật → hiểu câu hỏi → trả lời. Tất cả chạy trên ESP32-S3.

## 1. Cần có

- ESP32-S3 có PSRAM Octal (N16R8 / N8R8) + camera OV7670 (đấu chân như baseline BTC, xem [`README_baseline_BTC.md`](README_baseline_BTC.md#phần-cứng))
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/) ≥ 5.1 (đội dùng v6.1)

## 2. Build và nạp

```
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```

## 3. Kết nối WiFi

1. Sau khi nạp, ESP phát WiFi **`ESP32S3-INFER`**, mật khẩu **`12345678`**. Kết nối vào rồi mở http://192.168.4.1/
2. Muốn dùng WiFi khác: mở http://192.168.4.1/wifi, nhập tên + mật khẩu WiFi. ESP khởi động lại và vào mạng đó.
3. Địa chỉ IP mới in trong log `idf.py monitor`, hoặc mở http://esp32s3-infer.local/

## 4. Sử dụng

**Trên web:** mở `http://<IP>/` → nhập câu hỏi → bấm **Run camera**. Web hiện ảnh camera, khung đồ vật,
intent, câu trả lời và thời gian xử lý.

**Gọi API:**

```
curl -X POST http://<IP>/run_cam --data "co bao nhieu chai"
```

Trả về JSON, trong đó `answer` là câu trả lời, `intent` là loại câu hỏi, `dets` là danh sách đồ vật.

| Loại câu hỏi | Ví dụ | Trả lời |
|---|---|---|
| COUNT | co bao nhieu chai | Có 2 chai. |
| EXIST | co chuot khong | Có chuột trong ảnh. |
| LIST | tren ban co nhung gi | Có 3 loại đối tượng: 1 chai, 1 cốc, 2 sách. |
| LOCATION | dien thoai o dau | Điện thoại nằm bên trái. |
| OUT_OF_SCOPE | hom nay troi mua khong | Câu hỏi nằm ngoài phạm vi. |

Xem camera trực tiếp: `http://<IP>:81/stream`

## 5. Model và cấu hình

| File | Nội dung |
|---|---|
| `main/models/det_model.cc` | detect YOLO P2P4 INT8 (191.145 tham số), tạo bằng [notebook 09](../chung_ket/notebooks/09_xuat_detect_int8_cc.ipynb) |
| `main/models/textcnn_model.cc`, `vocab.cc` | intent TextCNN INT8 (186.117 tham số), tạo bằng [notebook 10](../chung_ket/notebooks/10_intent_textcnn.ipynb) |
| `main/model_config.h` | ngưỡng từng lớp, NMS |
| `main/logic.c` | tìm tên đồ vật trong câu hỏi, tạo câu trả lời |
