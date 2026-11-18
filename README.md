# Edge AI Challenge 2026 Lần 2 – Vòng loại

Bài dự thi [EDGE AI CHALLENGE 2026 LẦN 2 – Qualifiers](https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers):
phát hiện 7 loại đồ vật trên ảnh 320×240 và phân loại ý định câu hỏi (5 intent).

| | |
|---|---|
| **Bài nộp chính thức** | `submission_kd_ep232.csv`, **public 99,021** |
| Detection | YOLO P2/P3/P4 tự thiết kế, **191.145 tham số**, train từ đầu, distill từ thầy cũng train từ đầu; mAP@0.5 val 0,982 |
| Intent | TextCNN 18.213 tham số, accuracy 1,0 trên 20% câu train giữ lại |
| Chạy lại / kiểm tra | **[`pipeline/`](pipeline/README.md)**: notebook chạy từ trên xuống, kèm trọng số đã nộp |

## Kiểm tra nhanh cho BTC

Mọi thứ cần để kiểm tra bài nộp nằm trong **`pipeline/`** ([hướng dẫn chi tiết](pipeline/README.md)):

| Notebook | Detection | Bài nộp | mAP val |
|---|---|---|---|
| `pipeline/pipeline.ipynb` | trò + distill (**bài chính thức**) | 99,021 | 0,9819 |
| `pipeline/pipeline_student.ipynb` | trò tự học, không distill | 98,932 | 0,9812 |

- `MODE = "infer"` (mặc định): nạp trọng số đã nộp trong `pipeline/weights/`, dự đoán 4.000 ảnh test, ghép intent,
  ghi `submission.csv`. Mất khoảng 3 phút trên Kaggle T4. Đã chạy thử trên Kaggle: mAP val đúng 0,9819, intent trùng
  4.000/4.000 câu với bài đã nộp. Chạy trên máy dùng để train thì file ra giống hệt từng byte bài đã nộp.
- `MODE = "train"`: train lại từ đầu. Đã chạy thử trên Kaggle T4 với `pipeline.ipynb`: mất 3,2 giờ, mAP val 0,9828.
  Huấn luyện trên GPU không cho kết quả giống từng bit, nên mAP val dao động khoảng ±0,002 giữa các lần chạy.

Trên Kaggle: Import Notebook -> Add Input cuộc thi -> bật Internet (để `pip install ultralytics==8.4.199`) -> đặt
`REPO_URL` thành link repo này (hoặc thêm thư mục `pipeline/` làm Dataset) -> Run All. Kết quả ở
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

Kết quả: mAP val tăng từ 0,972 lên 0,982, toy_car tăng nhiều nhất (+0,026). Điểm public tăng từ 98,232 lên 99,021.

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
| 18/11 | **`yolo-kd`: YOLO P2P4 + distill (bài chính thức)** | **0,982** | **99,021** |
| 18/11 | `pipe-train`: chạy lại `yolo-kd` bằng notebook trên Kaggle | 0,983 | 98,937 |

Điểm = 0,7 × mAP@0.5 detection × 100 + 30 × accuracy intent.

## Cấu trúc repo

```
pipeline/                  ★ gói kiểm tra cho BTC: 2 notebook, trọng số đã nộp, cấu hình model (xem pipeline/README.md)
yolo_tiny.py               script gốc đã train các model YOLO (yolo-teacher, yolo-student, yolo-kd) trên máy cá nhân
yolo_cfg/                  cấu hình YOLO: student_p2p4.yaml (bài nộp), teacher_p2p4.yaml (thầy)
kaggle_runs/
  results.csv              ★ bảng kết quả của 46 thí nghiệm (mô tả cột ở dưới)
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
