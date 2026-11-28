# Edge AI Challenge 2026 Lần 2 – Bài nộp Vòng Chung Kết

> **Ban giám khảo xem [`chung_ket/README.md`](chung_ket/README.md)**: thứ tự notebook train end-to-end
> (`chung_ket/notebooks/01…10`), dữ liệu + pretrained đã dùng, trọng số thi đấu, cách build/nạp firmware.
>
> - Firmware thi đấu: [`firmware/`](firmware) (model đã xuất `.cc` trong `firmware/main/models/`)
> - Detect thi đấu: **cam4B** (YOLO P2P4 INT8, 191.145 tham số) · Intent: **TextCNN v5** INT8 (186.117 tham số)
> - Phần dưới đây là ghi chép vòng loại và quá trình thử nghiệm (giữ để tham khảo; bảng firmware bên dưới có thể
>   ghi model cũ: cấu hình đúng của bài thi nằm trong `chung_ket/README.md` và `firmware/main/model_config.h`).

---

# Edge AI Challenge 2026 Lần 2 – Vòng loại

Bài dự thi [EDGE AI CHALLENGE 2026 LẦN 2 – Qualifiers](https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers):
phát hiện 7 loại đồ vật trên ảnh 320×240 và phân loại ý định câu hỏi (5 intent).

| | |
|---|---|
| **Bài nộp chính thức** | bài `yolo-kd300` (nộp 18/11/2026 21:01), **public 99,061** |
| Detection | YOLO P2/P3/P4 tự thiết kế, **191.145 tham số**, train từ đầu, distill từ thầy cũng train từ đầu; mAP@0.5 val 0,982 |
| Intent | TextCNN 18.213 tham số, accuracy 1,0 trên 20% câu train giữ lại |
| Chạy lại / kiểm tra | **[`pipeline/`](pipeline/README.md)**: notebook chạy từ trên xuống, kèm trọng số đã nộp |
| Vòng chung kết | ESP32-S3 + camera OV7670, xem [mục Vòng chung kết](#vòng-chung-kết-esp32-s3--camera-ov7670) |

## Kiểm tra nhanh cho BTC

Mọi thứ cần để kiểm tra bài nộp nằm trong **`pipeline/`** ([hướng dẫn chi tiết](pipeline/README.md)):

| Notebook | Detection | Bài nộp | mAP val |
|---|---|---|---|
| `pipeline/pipeline.ipynb` | trò + distill (**bài chính thức**) | 99,061 | 0,9827 |
| `pipeline/pipeline_student.ipynb` | trò tự học, không distill | 98,932 | 0,9812 |

- `MODE = "infer"` (mặc định): nạp trọng số đã nộp trong `pipeline/weights/`, dự đoán 4.000 ảnh test, ghép intent,
  ghi `submission.csv`. Mất khoảng 3 phút trên Kaggle T4. Chạy trên máy dùng để train thì file ra giống hệt từng byte
  bài đã nộp. Trên Kaggle T4: cùng mAP val, intent trùng 4.000/4.000 câu, score box chỉ lệch ở chữ số cuối do khác GPU.
- `MODE = "train"`: train lại từ đầu (thầy 200 epoch, trò distill 300 epoch, ước tính 3,7 giờ trên Kaggle T4). Đã chạy
  thử trên Kaggle với trò dừng ở epoch 232: mất 3,2 giờ, mAP val 0,9828.
  Huấn luyện trên GPU không cho kết quả giống từng bit, nên mAP val dao động khoảng ±0,002 giữa các lần chạy.

Trên Kaggle: Import Notebook -> Add Input cuộc thi -> bật Internet (để `pip install ultralytics==8.4.199` và tự clone
repo này lấy trọng số, `REPO_URL` đã điền sẵn) -> Run All. Kết quả ở
`/kaggle/working/submission.csv`.

## Ràng buộc vòng loại

| Hạng mục | Giới hạn | Bài nộp |
|---|---|---|
| Đầu vào | mỗi chiều ≤ 320×240 | 320×240, ảnh gốc đưa thẳng vào model |
| Số tham số detector | ≤ 200.000 | 191.145 |
| Dung lượng model | ≤ 1,5 MB | 747 KB (float32) |
| Lượng tử INT8 | không bắt buộc ở vòng loại (BTC thông báo 11/11/2026) | model float |
| Pretrained / dữ liệu ngoài | không | không dùng |
| Ensemble / TTA | không | 1 model, 1 lần chạy mỗi ảnh |

Notebook tự `assert` số tham số, dung lượng và kích thước đầu ra trước khi dự đoán.

## Giải pháp

**Detection.** Model đầu tiên của đội là CenterNet + MobileNetV2 (TensorFlow, 196k tham số). Sau khoảng 40 thí
nghiệm, model này đạt tối đa mAP val 0,972 (public 98,232). Phân tích lỗi cho thấy một nửa số lỗi nằm ở vật nhỏ hơn
24 px. Đội chuyển sang YOLO11 (Ultralytics) sửa lại cho vật nhỏ:

- đầu ra **P2/4, P3/8, P4/16**, bỏ P5: lưới 80×60 + 40×30 + 20×15 = 6.300 anchor. Stride lớn nhất là 16 nên ảnh
  320×240 không phải đệm;
- **SCDown** (của YOLOv10) khi giảm kích thước để dồn tham số sang P2/P3; `reg_max` 4; bỏ attention C2PSA;
- **distill** từ một thầy 2,58 triệu tham số cùng bố cục anchor, train từ đầu trên dữ liệu BTC: BCE theo xác suất lớp
  của thầy và KL giữa phân bố DFL của thầy và trò, ở mọi anchor.

Kết quả: mAP val tăng từ 0,972 lên 0,983, toy_car tăng nhiều nhất (+0,029). Điểm public tăng từ 98,232 lên 99,061.

**Intent.** Chuẩn hóa câu -> tách từ -> TextCNN (Embedding 32, Conv1D kernel 2/3/4, GlobalMaxPool, Dense 5). Lúc train
sinh thêm câu bằng cách đổi tên vật (tên lấy từ `intent_train.csv`). Lúc dự đoán chỉ chạy TextCNN.

Chi tiết kiến trúc, loss và cấu hình train: [`pipeline/README.md`](pipeline/README.md).

## Lịch sử điểm public

| Ngày | Bài | mAP val | Public |
|---|---|---|---|
| 09/11 | CNN đầu tiên (TF) | | 86,378 |
| 12/11 | `A45-float`: CenterNet + MobileNetV2 alpha 0.45 | 0,960 | 97,214 |
| 14/11 | `final-s4`: thêm tầng stride 4 | (train cả 6.000 ảnh) | 97,781 |
| 16/11 | `final-ep300`: LR 3e-3, 300 epoch | (train cả 6.000 ảnh) | 98,217 |
| 17/11 | `final-ep400`: 450 epoch, bài TF tốt nhất | (train cả 6.000 ảnh) | 98,232 |
| 18/11 | `yolo-student`: YOLO P2P4 không distill | 0,981 | 98,932 |
| 18/11 | `yolo-kd`: YOLO P2P4 + distill, dừng ở epoch 232 (máy hết RAM) | 0,982 | 99,021 |
| 18/11 | `pipe-train`: chạy lại `yolo-kd` bằng notebook trên Kaggle | 0,983 | 98,937 |
| 18/11 | **`yolo-kd300`: YOLO P2P4 + distill đủ 300 epoch (bài chính thức)** | **0,983** | **99,061** |

Điểm = 0,7 × mAP@0.5 detection × 100 + 30 × accuracy intent.

## Vòng chung kết (ESP32-S3 + camera OV7670)

Cập nhật 27/11/2026. Firmware ở [`firmware/`](firmware/README.md), công cụ dữ liệu ở [`tools/`](tools/README.md),
các lần train trên Kaggle ở `kaggle_runs/cam*`, `kaggle_runs/gen*`.

### Firmware đang nạp

| | |
|---|---|
| Detector | **cam3R3B** (`kaggle_runs/cam3R3B`): YOLO P2P4 ~160k tham số, INT8, tầm với mức P4 ×1.5 |
| Ngưỡng conf | bottle **0.35**, cup **0.65**, mouse **0.70**, phone / book / box / toy_car 0.45 |
| NMS | **2 tầng:** IoU **0.30 theo lớp** + **0.60 khác lớp** (`YOLO_CROSS_NMS_IOU`, bỏ 2 nhãn đè khít lên cùng 1 vật) |
| LOCATION | lớp được hỏi không có box đủ ngưỡng thì lấy box điểm cao nhất ≥ **0.10** (`LOC_FLOOR`) |

NMS trên cam3R3B (độ đúng câu trả lời, ngưỡng conf như trên):

| NMS | BTC mẫu | real55 | khăn200 | khăn1075 |
|---|---|---|---|---|
| **2 tầng 0.30 + 0.60 (đang dùng)** | **0.533** | 0.434 | 0.614 | 0.605 |
| gộp lớp 0.45 (bản trước) | 0.532 | 0.434 | 0.614 | 0.603 |
| theo lớp 0.45, không tầng 2 | 0.528 | 0.475 | 0.627 | 0.615 |

Theo lớp không tầng 2 cao hơn chủ yếu ở LOCATION, nhưng hay để 2 nhãn khác lớp trên 1 vật nên đội không dùng.

`cam3R3A_merge` (`kaggle_runs/cam3R3A_merge`, commit bd01130) **không dùng**: model train trên 1.075 ảnh khăn trắng,
tức chính bộ test khăn200/khăn875. Trên ảnh chưa thấy, model này không hơn R3B (cấu hình của bản đó: real55 0.423,
BTC mẫu 0.535; R3B: 0.434 / 0.533). Điểm khăn200 0.731 của nó không dùng để so được.

Cấu hình nằm trong `firmware/main/model_config.h` (và `YOLO_CROSS_NMS_IOU` trong `firmware/main/logic.h`). Web/API đổi tạm khi chạy được: `?conf=` (mọi lớp), `&iou=`,
`&agn=`, `&loc=`.

### Bộ test (không bao giờ dùng để train)

| Bộ | Số ảnh | Ghi chú |
|---|---|---|
| **khăn200** (`data/cam_new_test`) | 200 | camera thật, nền khăn trắng, giống chung kết nhất, đã sửa nhãn tay |
| **khăn875** (`data/cam_new_train`) | 875 | cùng đợt chụp, nhãn đã sửa tay (bản nháp cũ ở `labels_prelabel/`) |
| khăn1075 (`outputs/khan1075`) | 1075 | 200 + 875 gộp |
| real55 | 55 | camera thật nền bàn (đợt 26/11) |
| BTC mẫu | 9 | ảnh mẫu thật của BTC |

Chấm bằng `python tools/eval_round.py <bản...>`: tải output Kaggle, xuất INT8, chấm **độ đúng câu trả lời** bằng đúng
cấu hình firmware (`CFG`, `CFG_CLS` trong script, phải sửa theo mỗi khi đổi `model_config.h`).

### Kết quả (độ đúng câu trả lời, cấu hình firmware hiện tại)

| Bản | BTC mẫu | real55 | khăn200 | khăn875 | khăn1075 |
|---|---|---|---|---|---|
| **cam3R3B (đang nạp)** | **0.531** | **0.434** | 0.614 | **0.601** | **0.603** |
| cam3R3A | 0.507 | 0.397 | 0.623 | 0.595 | 0.600 |
| cam3R2B | 0.519 | 0.402 | 0.621 | 0.595 | 0.599 |

mAP50 trên ảnh thật (real55) theo lớp, R2B → R3B: bottle 0.74 → 0.82, phone 0.77 → 0.95, mouse 0.44 → 0.66,
book 0.51 → 0.38.

Các thử nghiệm ngưỡng và NMS trên cam3R2B (khăn200 / khăn875 / BTC mẫu):

| Cấu hình | khăn200 | khăn875 | BTC mẫu |
|---|---|---|---|
| NMS 0.30 theo lớp, bottle 0.30 | 0.618 | 0.600 | 0.552 |
| NMS 0.30 theo lớp, bottle 0.35 | 0.619 | 0.601 | 0.537 |
| NMS 0.30 theo lớp, bottle 0.40 | 0.618 | 0.602 | 0.530 |
| NMS 0.30 / 0.45 / 0.60 / 0.70 gộp lớp, bottle 0.35 | 0.613 / 0.615 / 0.611 / 0.607 | 0.591 / 0.587 / 0.581 / 0.580 | 0.527 / 0.527 / 0.533 / 0.533 |
| **NMS 0.45 gộp lớp, bottle 0.35, cup 0.65, mouse 0.70 (đang dùng)** | **0.621** | **0.595** | **0.519** |

NMS theo lớp cho điểm cao hơn một chút nhưng hay ra 2 label khác lớp đè lên cùng một vật, nên đội chọn gộp lớp.

Độ khớp kích thước box (cam3R3B, khăn1075, box đoán / nhãn, trung vị): mọi lớp 1.00–1.08, IoU trung bình 0.75–0.87.
Box cup nhỏ hơn cái cốc chỉ là trường hợp lẻ (~5% số cốc có IoU < 0.5), không phải lỗi hệ thống.

### Dữ liệu train của cam3R3B (~27.300 ảnh 320×240)

| Nguồn | Số ảnh | Nội dung |
|---|---|---|
| Dán vật (gen5, `kaggle_runs/gen5`) | 12.000 | 6.000 ảnh dán + bản mô phỏng OV7670; 5–9 vật mỗi ảnh, xoay mọi hướng |
| Roboflow | 10.508 | 60% các bộ bottle, cup, book, box, mouse, phone, toy car, Hot Wheels; nhãn thiếu điền bằng YOLO11x, zoom-out, mô phỏng camera |
| Ảnh BTC 3D | 4.800 | 80% train BTC, chỉ bản mô phỏng OV7670 |

- Vật cắt theo mask để dán: COCO 2017 + LVIS (bottle 4022, cup 4041, phone 2926, box 2221, book 1713, mouse 945),
  Open Images 8.714, TACO 655 (chai nhựa trong, chai thủy tinh, cốc, hộp carton), Hot Wheels 2.503.
- gen5: ~60% ảnh nền giấy A0 trắng; ~60% chuột có dây; ~15% ảnh có dây gây nhiễu; 40% sách là sách tự tạo bìa (60% bìa
  xanh); trọng số dán mouse 1.8, book 1.5, bottle 1.5, box 1.2, cup 1, phone 1, toy_car 0.8.
- Train: thầy cam3 train tiếp 15 epoch; trò P2P4 khởi tạo từ trò cam3, 45 epoch, distill hệ số 2.0.
- Val chọn checkpoint: 1.200 ảnh BTC val + 460 ảnh Roboflow val.
- **Không có ảnh camera OV7670 thật nào trong train**, góc nhìn thẳng từ trên xuống rất ít, khoảng cách 85 cm chưa
  được kiểm riêng.

### Vì sao bottle vẫn thấp (cam3R3B, 338 chai trong khăn1075)

Bắt đúng 64% số chai. Chai bị sót có điểm bottle trung vị 0.23, nên hạ ngưỡng không cứu được. Có 104 lần báo chai sai.
Lưu ý: khăn1075 gồm nhiều khung liên tiếp của cùng một cảnh, nên một vật có thể bị đếm nhiều lần.

| Nhóm | Số chai | Bắt được |
|---|---|---|
| Đứng, to, không bị che | 161 | 73% |
| Nằm ngang | 89 | 81% |
| **Chồng lên / dựa vào vật khác** | 61 | **46%** |
| **Nhỏ (cạnh dài < 40 px)** | 27 | **7%** |

Nguyên nhân:
1. **Che khuất:** `coco_paste.py` không cho vật bị che quá 15%, chỉ ~20% vật được dán đè, nên model hầu như chưa thấy
   chai đứng trên hoặc dựa vào hộp. NMS gộp lớp còn hay để box hộp xoá box chai.
2. **Chai nhựa trong suốt trên nền trắng:** gần như không có tương phản. Train chỉ có 655 chai TACO (phần lớn bị bóp
   méo); chai COCO đa số là chai thủy tinh hoặc chai đục, chụp ngang.
3. **Chai nhỏ ở xa:** vật COCO phải có cạnh ≥ 40 px (`--min-px 40`) mới được chọn.
4. **Báo sai ở cuộn dây cáp đen** (~40% số lần báo sai đã xem): gen5 chỉ có dây kéo dài, không có dây cuộn tròn.
5. **Nhãn còn sót:** ~10–15% "báo sai" là chai thật chưa gán nhãn trong khăn875; vài nhãn bottle đáng ngờ (vd ảnh 507).

Ngoài ra ~9% chai bị nhận thành cup (real55: 16%). Góc nhìn thẳng từ trên xuống cũng kém, vì chai chỉ còn là một hình
tròn giống miệng cốc, mà train hầu như không có góc này.

### Việc tiếp theo (đề xuất gen6)

1. Tăng che khuất khi dán (`--occlude` 0.2 → 0.4, che tối đa 15% → 40%), thêm kiểu chai dựa hoặc đứng trên hộp/sách.
2. Thêm chai trong suốt (Roboflow plastic/water bottle, tăng tỉ lệ TACO), thêm chai nhìn từ trên xuống.
3. Dây cuộn tròn làm nền (dán vào, không gán nhãn).
4. Hạ `--min-px`, thêm vật cỡ 20–40 px.
5. Sửa các chai chưa gán nhãn trong khăn875 (người sửa tay).
6. Chụp bộ ảnh mới bằng camera ESP ở 85 cm (cả góc thẳng xuống), tách riêng khỏi bộ khăn: dùng để test khoảng cách,
   và nếu đội đồng ý thì một phần dùng để train.

## Cấu trúc repo

```
pipeline/                  ★ gói kiểm tra cho BTC: 2 notebook, trọng số đã nộp, cấu hình model (xem pipeline/README.md)
yolo_tiny.py               script gốc đã train các model YOLO (yolo-teacher, yolo-student, yolo-kd, yolo-kd300) trên máy cá nhân
yolo_cfg/                  cấu hình YOLO: student_p2p4.yaml (bài nộp), teacher_p2p4.yaml (thầy)
kaggle_runs/
  results.csv              ★ bảng kết quả của 47 thí nghiệm (mô tả cột ở dưới)
  <mã run>/                mỗi thí nghiệm 1 thư mục: script đúng như lúc chạy (edgeai_kaggle.py hoặc pipeline.ipynb),
                           kernel-metadata.json (cấu hình kernel Kaggle), output/*.log (log chạy)
requirements.txt
data/                      dữ liệu cuộc thi (không có trên git)
```

Model, file submission và trọng số của từng run không đưa lên git (nặng, tạo lại được). Trọng số của các bài đã nộp
nằm trong `pipeline/weights/`.

### `kaggle_runs/results.csv`

Mỗi dòng là một thí nghiệm. mAP val tính trên cùng 1.200 ảnh val (tách 20% từ train, seed 42), cùng một hàm mAP@0.5.

| Cột | Ý nghĩa |
|---|---|
| `ma` | mã run, trùng tên thư mục trong `kaggle_runs/` (riêng `yolo-*` train trên máy, script là `yolo_tiny.py`) |
| `thay_doi` | thay đổi so với mốc trước |
| `tham_so` | số tham số detector |
| `s_per_epoch`, `epoch_tot_nhat` | thời gian mỗi epoch (giây), epoch có mAP val tốt nhất |
| `map_float`, `map_int8` | mAP@0.5 val của model float / INT8 (INT8 chỉ có ở các run đầu, trước khi BTC bỏ yêu cầu lượng tử) |
| `ap_<lớp>` | AP@0.5 val từng lớp |
| `tflite_kb`, `ops` | dung lượng TFLite và danh sách op (chỉ các run đầu) |
| `rang_buoc_ok`, `submission_ok` | kiểm tra ràng buộc / file submission hợp lệ |
| `gpu`, `gio_gpu` | phần cứng, số giờ GPU (Kaggle/Colab; run trên máy cá nhân để trống) |
| `ghi_chu` | kết luận: GIỮ / KHÔNG GIỮ và lý do, điểm public nếu đã nộp |

Run có mAP val ghi là "GIẢ" là run train trên cả 6.000 ảnh (val nằm trong train), chỉ so được qua điểm public.

## Khai báo theo thể lệ

- **Bài nộp chính thức** không dùng pretrained, dữ liệu ngoài, ensemble hay TTA. Thầy dùng để distill được train từ
  đầu trên dữ liệu BTC và chỉ dùng lúc train.
- **Một số thí nghiệm cũ** trong `kaggle_runs/` (không nộp làm bài chính thức) có thử pretrained: `pre35` (backbone
  MobileNetV2 ImageNet) và `kd` (distill từ YOLO11s). Hai hướng này bị bỏ khi đội biết thể lệ chỉ cho dùng pretrained
  ở vòng chung kết. Kết quả vẫn giữ trong `results.csv` để minh bạch.
- **Mã nguồn có ảnh hưởng:** [Ultralytics](https://github.com/ultralytics/ultralytics) 8.4.199 (AGPL-3.0) cho các khối
  YOLO11/YOLOv10, loss và vòng lặp train. Kiến trúc P2P4, loss distill, hàm dự đoán, hàm mAP và TextCNN do đội tự viết.
- **Môi trường:** các model YOLO đã nộp được train trên máy cá nhân (RTX 5060 Laptop). Các thí nghiệm TF (và TextCNN của
  bài nộp) được train trên Kaggle T4 hoặc Colab. Notebook trong `pipeline/` đã chạy lại được trên Kaggle T4.
