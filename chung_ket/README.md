# EdgeAI Challenge 2026 – Bài nộp Vòng Chung Kết

Hỏi đáp về đồ vật chạy hoàn toàn trên **ESP32-S3 + camera OV7670**.

```
ảnh 320×240 ──► YOLO P2P4 INT8 (191.145 tham số) ──► box + lớp ─┐
                                                               ├─► câu trả lời
câu hỏi ──► TextCNN INT8 (186.117 tham số) ──► intent ─────────┘
```

## Nội dung

| Thư mục | Nội dung |
|---|---|
| `notebooks/` | notebook huấn luyện, chạy theo thứ tự 01 → 10 |
| `weights/` | trọng số dùng thi đấu |
| `data/cam_real_ov7670/` | 1.222 ảnh camera thật đội tự thu + nhãn |
| `DU_LIEU_VA_PRETRAINED.md` | danh sách dữ liệu và mô hình pretrained (kèm link) |
| `../firmware/` | firmware thi đấu (model `.cc` trong `main/models/`) |

## Thứ tự notebook

| # | Notebook | Việc |
|---|---|---|
| 01–04 | `du_lieu_*` | tạo dữ liệu tổng hợp |
| 05 | `detect_cam1` | train thầy (YOLO11s pretrained COCO) |
| 06 | `detect_cam3C` | train trò từ đầu, distill từ thầy |
| 07 | `detect_R3B` | train tiếp thầy + trò |
| 08 | `detect_cam4B_FINAL` | train tiếp với ảnh camera thật → **model thi đấu** |
| 09 | `xuat_detect_int8_cc` | INT8 → `det_model.cc` |
| 10 | `intent_textcnn` | train intent → INT8 → `textcnn_model.cc` |

Notebook 01–08 chạy trên Kaggle (ô đầu mỗi notebook ghi dataset cần gắn). 09–10 chạy trên máy tại thư mục gốc repo.

## Build firmware

```
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash
```
