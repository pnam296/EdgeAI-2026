# EdgeAI Challenge 2026 – Bài nộp Vòng Chung Kết

Hệ thống hỏi đáp về đồ vật chạy **hoàn toàn trên ESP32-S3 + camera OV7670**:

```
OV7670 320×240 ──► YOLO P2P4 INT8 (191.145 tham số, 1 lần chạy/ảnh) ──► NMS 2 tầng ──► box + lớp ─┐
                                                                                                  ├─► câu trả lời tiếng Việt
câu hỏi ──► bỏ dấu, chuẩn hóa ──► 24 token ──► TextCNN INT8 (186k tham số) ──► intent ──────────┘
```

| Hạng mục | Giới hạn BTC | Bài nộp |
|---|---|---|
| Detect: số tham số | ≤ 200.000 | **191.145** |
| Detect: file INT8 | ≤ 1,5 MB | **328 KB** (`firmware/main/models/det_model.tflite`) |
| Intent: số tham số / file INT8 | ≤ 200.000 / ≤ 1,5 MB | **186.117 / 195 KB** |
| Đầu vào | ≤ 320×240 | 320×240 |
| Số lần chạy model mỗi ảnh | 1 (không ensemble, không TTA) | 1 |

Lớp: `bottle cup phone book mouse box toy_car` · Intent: `COUNT EXIST LIST LOCATION OUT_OF_SCOPE`

## 1. Thứ tự chạy notebook (end-to-end)

Thư mục `notebooks/`. Chạy **đúng thứ tự số**. Notebook 01–08 chạy trên Kaggle: mỗi notebook ghi ở ô đầu các
dataset cần gắn (Add Input) và output của notebook trước cần gắn. Notebook 09–10 chạy trên máy từ thư mục gốc repo.

| # | Notebook | Làm gì | Máy | Thời gian | Kết quả |
|---|---|---|---|---|---|
| 01 | `01_du_lieu_oicrops` | cắt vật từ Open Images làm kho vật | Kaggle CPU | ~30 phút | kho vật oicrops |
| 02 | `02_du_lieu_tacocrops` | cắt vật theo mask từ TACO (chai/cốc/hộp) | Kaggle CPU | ~15 phút | kho vật tacocrops |
| 03 | `03_du_lieu_gen4` | sinh ảnh tổng hợp gen4 (Roboflow + dán vật COCO/LVIS/OI) | Kaggle CPU | ~1 giờ | dữ liệu gen4 |
| 04 | `04_du_lieu_gen5` | sinh ảnh gen5: nền giấy A0 trắng giống chung kết, 5–9 vật/ảnh | Kaggle CPU | ~1 giờ | dữ liệu gen5 |
| 05 | `05_detect_cam1_thay_coco` | **thầy** YOLO11s P2P4 từ backbone COCO, 60 epoch | Kaggle T4 | ~3 giờ | thầy cam1 |
| 06 | `06_detect_cam3C` | thầy cam1 +30 epoch; **trò train từ đầu** 100 epoch + distill | Kaggle T4 | ~4 giờ | thầy + trò cam3C |
| 07 | `07_detect_R3B` | thầy +15, trò +45 epoch (KD 2.0) trên gen5 | Kaggle T4 | ~2,5 giờ | thầy + trò R3B |
| 08 | `08_detect_cam4B_FINAL` | **model thi đấu**: thầy +25, trò +45 epoch + 1222 ảnh camera thật | Kaggle T4 | ~3 giờ | `yolo_cam4B_last.pt` |
| 09 | `09_xuat_detect_int8_cc` | lượng tử INT8 → `det_model.cc` → chép vào firmware | máy, CPU | ~2 phút | `firmware/main/models/det_model.cc` |
| 10 | `10_intent_textcnn` | train TextCNN → INT8 → `textcnn_model.cc`, `vocab.cc` → chép vào firmware; chấm | máy, CPU | ~3 phút | `firmware/main/models/textcnn_model.cc` |

Muốn kiểm tra nhanh mà không train lại: bỏ qua 01–08, chạy 09 với trọng số có sẵn trong `weights/`.

**Tái lập:** train trên GPU không cho ra trọng số giống hệt từng byte giữa các lần chạy. Trọng số đúng của bài thi
nằm trong `weights/`; chạy lại notebook cho model tương đương.

## 2. Trọng số đi kèm (`weights/`)

| File | Là gì |
|---|---|
| `yolo_cam4B_last.pt` | trò detect thi đấu (float, trước INT8), 191.145 tham số |
| `yolo_cam4B_teacher.pt` | thầy dùng để distill (không nạp lên ESP) |
| `intent_textcnn.keras`, `intent_vocab.txt` | TextCNN intent thi đấu (float) + vocab |

## 3. Dữ liệu

| Dữ liệu | Nguồn | Dùng ở |
|---|---|---|
| Ảnh + nhãn BTC (train/val), câu hỏi `intent_train.csv` | BTC | 05–08, 10 |
| Ảnh mô phỏng giống BTC (4.800 ảnh) | tự sinh | 05–08 |
| Roboflow (bottle, cup, phone, book, mouse, box, hot wheels…) | Roboflow Universe (ngoài) | 03, 05–08 |
| COCO 2017 (`awsaf49/coco-2017-dataset`) | ngoài | 03, 06 |
| Open Images (cắt vật) | ngoài | 01 |
| TACO (`kneroma/tacotrashdataset`) | ngoài | 02 |
| Ảnh camera thật OV7670 (1.222 ảnh, tự chụp, tự gán nhãn và sửa tay) | đội | 08 |

Dataset Kaggle của đội gắn vào các notebook: `phngnam296/edgeai-cam-data`, `phngnam296/edgeai-cam-real`,
`phngnam296/edgeai-cam3-data`, `phngnam296/edgeai-tools3`, `phngnam296/edgeai-cam-real2`,
`dohuuminhtri/edgeai-cam3-all`, `dohuuminhtri/edgeai-hotwheels`, `dohuuminhtri/edgeai-tools3`.

## 4. Mô hình pretrained (khai báo)

| Model | Dùng để |
|---|---|
| `yolo11s.pt` (Ultralytics, COCO) | khởi tạo backbone thầy (05). Không nạp lên ESP |
| `yolo11x.pt` (Ultralytics, COCO) | gán nhãn bổ sung cho ảnh COCO khi tạo dữ liệu (06) |
| fastText tiếng Việt, rút gọn 32 chiều (`pipeline/weights/vi_fasttext_32d.npz`) | khởi tạo embedding TextCNN (10) |

Trò detect (model chạy trên ESP) có kiến trúc tự thiết kế (`yolo_cfg/student_p2p4.yaml`), không lấy trọng số
pretrained bên ngoài: train từ đầu ở 06 rồi train tiếp ở 07, 08.

## 5. Firmware (`../firmware`)

Đây là bản dùng để thi đấu. Model đã xuất sẵn trong `firmware/main/models/` (`det_model.cc`, `textcnn_model.cc`, `vocab.cc`).

```
# ESP-IDF (đã cài, export.ps1 / export.sh)
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```

Cấu hình chính (`firmware/main/model_config.h`, `logic.h`):
- NMS 2 tầng: cùng lớp IoU 0,30 + khác lớp IoU 0,60 (không để 2 nhãn trên 1 vật)
- Ngưỡng từng lớp: bottle 0,60 · cup 0,50 · phone 0,70 · book 0,45 · mouse 0,70 · box 0,65 · toy_car 0,65
- LOCATION: trái / giữa / phải theo tâm box; tên vật tìm bằng từ khóa (được BTC cho phép); câu không có vật nào trong 7 lớp → OUT_OF_SCOPE

## 6. Kết quả (độ đúng câu trả lời, mô phỏng đúng firmware)

| Tập | cam4B (thi đấu) |
|---|---|
| Ảnh camera thật giữ lại, nền khăn (100 ảnh) | 0,764 |
| Ảnh camera thật giữ lại, mới (25 ảnh) | 0,814 |

| Intent (hỏi đáp, gồm tìm tên vật) | Đúng |
|---|---|
| `intent_train.csv` BTC (6.000 câu) | 100% |
| `intent_exam.csv` (163 câu) | 98% |
| `intent_hard_exam.csv` (114 câu khó) | 84% |

## Cấu trúc repo

```
chung_ket/        <- BÀI NỘP: notebook, trọng số, README này
firmware/         <- firmware ESP32-S3 thi đấu (main/, models/*.cc, tools/ để kiểm tra)
tools/            <- script dùng chung (train_intent.py, yolo_reach.py, sinh dữ liệu…)
yolo_cfg/         <- yaml kiến trúc thầy / trò
pipeline/         <- bài vòng loại (giữ để tham khảo; weights/ chứa fastText cho intent)
archive/          <- các thử nghiệm cũ (không cần để chạy bài nộp)
```
