# EdgeAI Challenge 2026 – Vòng Chung Kết

Hỏi đáp về đồ vật chạy hoàn toàn trên **ESP32-S3 + camera OV7670**.

## Bắt đầu từ đâu

| Bước | Làm gì | Hướng dẫn |
|---|---|---|
| 1 | Nạp firmware lên ESP32-S3 và chạy thử | [`firmware/README.md`](firmware/README.md) |
| 2 | Xem tổng quan bài nộp | [`chung_ket/README.md`](chung_ket/README.md) |
| 3 | Xem dữ liệu và pretrained đã dùng | [`chung_ket/DU_LIEU_VA_PRETRAINED.md`](chung_ket/DU_LIEU_VA_PRETRAINED.md) |
| 4 | Train lại model (notebook 01 → 10) | [`chung_ket/notebooks/`](chung_ket/notebooks) |

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `firmware/` | firmware thi đấu (model đã xuất `.cc` trong `main/models/`) |
| `chung_ket/` | notebook, trọng số, dữ liệu tự thu, danh sách dữ liệu + pretrained |
| `tools/`, `yolo_cfg/` | script và cấu hình kiến trúc dùng trong notebook |
| `pipeline/` | bài vòng loại |
| `archive/` | thử nghiệm cũ |
