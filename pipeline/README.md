# Edge AI Challenge 2026 – Vòng loại: pipeline của đội

Phát hiện đồ vật (**YOLO P2/P3/P4 tự thiết kế, 191.145 tham số, train từ đầu**) và phân loại ý định câu hỏi
(**TextCNN**) cho cuộc thi
[EDGE AI CHALLENGE 2026 LẦN 2 – Qualifiers](https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers).

| | |
|---|---|
| Bài nộp chính thức | **public 99,021** (`submission_kd_ep232.csv`, nộp 18/11/2026) |
| Detection | mAP@0.5 val **0,9819** (1.200 ảnh tách từ train), model float, 1 lần chạy/ảnh |
| Intent | accuracy 1,0 trên 20% câu train giữ lại |
| Tái lập | `MODE = "infer"` trên máy đã train: file **giống hệt từng byte** bài đã nộp (MD5 `4d21be92a28705e98c736b85f671517d`). Trên Kaggle T4: cùng mAP val 0,9819, intent trùng 4.000/4.000, score box chỉ lệch ở số lẻ do khác GPU |

```
ảnh 320×240 RGB ──► YOLO P2P4 (191k tham số) ──► 6.300 anchor × (4 box + 7 lớp) ──► NMS 0,7 ──► ≤ 100 box
câu hỏi ──► chuẩn hóa ──► 24 token ──► TextCNN (18k tham số) ──► 1 trong 5 intent
```

- **Lớp:** `bottle cup phone book mouse box toy_car`
- **Intent:** `COUNT EXIST LIST LOCATION OUT_OF_SCOPE`

## Mục lục
- [Ràng buộc vòng loại](#ràng-buộc-vòng-loại)
- [Chạy lại](#chạy-lại)
- [Pipeline](#pipeline)
- [Mô hình](#mô-hình)
- [Kết quả](#kết-quả)
- [Khai báo theo thể lệ](#khai-báo-theo-thể-lệ)
- [Cấu trúc](#cấu-trúc)

## Ràng buộc vòng loại

| Hạng mục | Giới hạn | Bài nộp | Kiểm tra trong notebook |
|---|---|---|---|
| Đầu vào | mỗi chiều ≤ 320×240 | 320×240, ảnh gốc đưa thẳng vào model (không đệm, không co) | ô 1.4 |
| Số tham số detector | ≤ 200.000 | **191.145** | ô 1.4 (`assert`) |
| Dung lượng model | ≤ 1,5 MB | **747 KB** (191.145 × 4 byte float32) | ô 1.4 (`assert`) |
| Lượng tử INT8 | không bắt buộc ở vòng loại (BTC thông báo 11/11/2026) | model float | |
| Pretrained / dữ liệu ngoài | không | dựng model từ yaml, `pretrained=False`, chỉ dùng dữ liệu BTC | |
| Ensemble / TTA | không | 1 model, chạy 1 lần mỗi ảnh | |

## Chạy lại

Có 2 notebook, cùng cấu trúc, chạy từ trên xuống:

| Notebook | Detection | Bài nộp | mAP val |
|---|---|---|---|
| `pipeline.ipynb` | trò + distill từ thầy (bài chính thức) | 99,021 | 0,9819 |
| `pipeline_student.ipynb` | trò tự học, không thầy, không distill | 98,932 | 0,9812 |

Chọn chế độ ở ô **Cấu hình**:

| `MODE` | Làm gì | Thời gian |
|---|---|---|
| `"infer"` (mặc định) | Nạp trọng số đã nộp trong `weights/`, chấm mAP val, dự đoán 4.000 ảnh test, ghép intent -> `submission.csv` | khoảng 2 phút có GPU |
| `"train"` | Train lại từ đầu -> TextCNN -> `submission.csv`. `pipeline.ipynb`: thầy 200 epoch -> trò distill (dừng ở epoch 232/300 như bài nộp). `pipeline_student.ipynb`: trò dừng ở epoch 290/300 như bài nộp | `pipeline.ipynb` đo trên Kaggle T4: 3,2 giờ (thầy 94 phút, trò 96 phút), mAP val 0,9828 |

`QUICK_TEST = True` chạy thử toàn bộ chế độ `train` với 200 ảnh, 2 epoch, xong trong vài phút.

### Kaggle

1. **New Notebook** -> File -> **Import Notebook** -> chọn `pipeline.ipynb`.
2. **Add Input** -> Competition `egde-ai-challenge-2026-lan-2-qualifiers`.
3. Lấy `weights/` và `yolo_cfg/`, chọn một trong hai cách:
   - Bật **Internet**, đặt `REPO_URL = "<link git của repo này>"` ở ô Cấu hình. Notebook sẽ tự clone.
   - Hoặc tải thư mục này lên làm Kaggle Dataset rồi **Add Input**. Notebook tự tìm trong `/kaggle/input`.
4. Accelerator **GPU T4**, Internet bật nếu môi trường chưa có `ultralytics` (notebook tự `pip install ultralytics==8.4.199`).
5. **Run All**. Kết quả nằm ở `/kaggle/working/submission.csv`. Trọng số và log train nằm ở `/kaggle/working/out/`.

### Máy cá nhân / Colab

```bash
pip install -r requirements.txt          # cài torch bản có CUDA theo hướng dẫn của pytorch.org nếu cần
kaggle competitions download -c egde-ai-challenge-2026-lan-2-qualifiers   # giải nén vào data/
jupyter notebook pipeline.ipynb
```

Notebook tự tìm dữ liệu (thư mục có `classes.txt`) trong `/kaggle/input`, `/content/data`, `data/` hoặc `../data/`.
Có thể đặt thẳng bằng `DATA_DIR`. Kết quả ghi vào `outputs/`.

> Huấn luyện YOLO trên GPU không cho ra kết quả giống hệt từng bit giữa các lần chạy, nên chế độ `train` cho mAP val
> lệch khoảng ±0,002 quanh 0,982. Chế độ `infer` cho cùng mAP val; trên GPU khác, vài score box có thể lệch ở chữ
> số cuối.

## Pipeline

| Phần | Nội dung | Sinh ra |
|---|---|---|
| Chuẩn bị | Cài `ultralytics==8.4.199`, tìm dữ liệu và thư mục pipeline | |
| 1.1 Dữ liệu | Chia train/val 80/20 (`SPLIT_SEED = 42`), chép sang cấu trúc thư mục Ultralytics; hàm mAP@0.5 (AP nội suy 101 điểm, trung bình 7 lớp) | `outputs/ds_split/` |
| 1.2 Mô hình | Loss distill `KDLoss`, hàm train (dựng model từ yaml), hàm dự đoán (ảnh 320×240 đưa thẳng vào model, NMS) | |
| 1.3 Train | Thầy `teacher_p2p4.yaml` 200 epoch -> trò `student_p2p4.yaml` distill, chọn `best.pt`/`last.pt` theo mAP val | `outputs/teacher/`, `outputs/student_kd/` |
| 1.4 Kiểm tra + dự đoán | `assert` số tham số, dung lượng, shape đầu ra; mAP val; dự đoán test | `outputs/det_test.json` |
| 2. Intent | TextCNN: train (holdout 20% + mô hình cuối) hoặc nạp `weights/intent_textcnn.keras`; dự đoán `intent_test.csv` | `outputs/intent_textcnn.keras` |
| 3. Submission | Ghép theo đúng thứ tự `sample_submission.csv`, tối đa 100 box/ảnh, score trong [0, 1], kiểm tra đủ dòng | `outputs/submission.csv` |

## Mô hình

### Detection: trò (bài nộp), `yolo_cfg/student_p2p4.yaml`

YOLO11 sửa lại cho vật nhỏ. Với model TF cũ của đội, một nửa số lỗi nằm ở vật nhỏ hơn 24 px.

- **Đầu ra P2/4, P3/8, P4/16**, bỏ P5. Lưới 80×60 + 40×30 + 20×15 = **6.300 anchor**. Stride lớn nhất là 16 nên ảnh
  320×240 chia hết, đưa thẳng vào model mà không phải đệm.
- **SCDown** (pointwise + depthwise, của YOLOv10) thay conv 3×3 khi giảm kích thước, để dồn tham số sang P2/P3.
  Số kênh P2/P3/P4 là 32/64/64.
- `reg_max` 4 (DFL nhỏ). Bỏ C2PSA (attention) cho vừa trần tham số.
- 191.145 tham số.

### Detection: thầy (không nộp), `yolo_cfg/teacher_p2p4.yaml`

Cùng bố cục 6.300 anchor, 2,58 triệu tham số, **train từ đầu trên dữ liệu BTC** (200 epoch, cùng tập train).
Thầy chỉ dùng trong lúc train trò, không chạy lúc dự đoán.

### Distill (`KDLoss`)

Loss = loss thường của Ultralytics (box IoU + DFL + lớp), cộng thêm ở **mọi anchor**:
- lớp: `BCE(logit của trò, sigmoid(logit của thầy))`
- box: `KL(phân bố DFL của thầy || của trò)`, nhân với độ tự tin cao nhất của thầy ở anchor đó

Cả hai chia cho tổng độ tự tin của thầy, hệ số `KD_CLS = KD_BOX = 1`.

Train: batch 64, ảnh 320, lịch 300 epoch mặc định của Ultralytics (`optimizer=auto` -> MuSGD lr 0,01, augment mặc
định có mosaic, tắt mosaic 10 epoch cuối).
Bài nộp là checkpoint `last.pt` ở **epoch 232**: lần chạy đó bị dừng ở epoch 232 vì máy hết RAM. `STOP_EPOCH = 232`
tái lập đúng điểm dừng này, đặt `STOP_EPOCH = 0` để chạy đủ 300 epoch.

### Intent: TextCNN (Keras, 18.213 tham số)

Chuẩn hóa (chữ thường, bỏ ký tự ngoài `a-z0-9`) -> tách từ theo dấu cách -> vocab 263 mục (gồm pad và từ lạ) -> 24 token ->
Embedding 32 + Conv1D 32 kênh (kernel 2, 3, 4) + GlobalMaxPool -> Dropout 0,3 -> Dense 5. Adam 3e-3, 20 epoch.

Lúc train có sinh thêm câu: với câu COUNT/EXIST/LOCATION có tên vật thuộc 7 lớp, thêm 1 câu đổi sang tên khác trong 7 lớp
(giữ nhãn) và 1 câu đổi sang vật ngoài 7 lớp (nhãn OUT_OF_SCOPE). 6.000 câu gốc thành 13.068 câu. Lúc dự đoán chỉ chạy
TextCNN, không dùng từ điển hay luật.

## Kết quả

mAP@0.5 trên 1.200 ảnh val (`SPLIT_SEED = 42`), cùng một hàm tính:

| Model | Tham số | bottle | cup | phone | book | mouse | box | toy_car | **mAP** | Public |
|---|---|---|---|---|---|---|---|---|---|---|
| Thầy (không nộp) | 2.580.073 | 0,990 | 0,990 | 0,985 | 0,982 | 0,985 | 0,990 | 0,986 | 0,987 | |
| Trò không distill, epoch 290 | 191.145 | 0,989 | 0,989 | 0,973 | 0,972 | 0,976 | 0,989 | 0,980 | 0,981 | 98,932 |
| **Trò + distill, epoch 232 (bài nộp)** | **191.145** | 0,987 | 0,989 | 0,975 | 0,972 | 0,978 | 0,989 | 0,983 | **0,982** | **99,021** |
| Model TF cũ của đội (CenterNet + MobileNetV2, run `lr3ep240`) | 196.307 | 0,981 | 0,986 | 0,966 | 0,962 | 0,967 | 0,989 | 0,954 | 0,972 | 98,232* |

\* điểm public của bài TF tốt nhất (cùng kiến trúc, train trên cả 6.000 ảnh, 450 epoch), không phải của đúng run `lr3ep240`.

Điểm = 0,7 × mAP@0.5 detection × 100 + 30 × accuracy intent.

## Khai báo theo thể lệ

- **Pretrained:** không dùng. Mọi model dựng từ yaml/code với trọng số ngẫu nhiên.
- **Dữ liệu ngoài:** không dùng. Chỉ dùng dữ liệu BTC.
- **Ensemble / TTA:** không dùng. Dự đoán bằng 1 model, 1 lần chạy mỗi ảnh.
- **Distillation:** thầy train từ đầu trên dữ liệu BTC, chỉ dùng lúc train.
- **Mã nguồn có ảnh hưởng:** [Ultralytics](https://github.com/ultralytics/ultralytics) 8.4.199 (AGPL-3.0): các khối
  C3k2, SPPF, Detect (YOLO11), SCDown (YOLOv10), `v8DetectionLoss`, vòng lặp train và NMS. Phần còn lại do đội tự
  viết: kiến trúc P2P4 (2 file yaml), `KDLoss`, hàm dự đoán, hàm mAP, TextCNN và cách sinh câu cho intent.
- **Sinh câu cho intent:** các tên vật dùng để thay thế đều lấy từ `intent_train.csv`. Chỉ dùng lúc train.
- **Môi trường:** detection của bài nộp được train trên máy cá nhân (RTX 5060 Laptop, Windows, PyTorch 2.11,
  Ultralytics 8.4.199). TextCNN được train trên Kaggle (T4). Notebook này chạy được trên Kaggle để BTC kiểm tra lại.

## Cấu trúc

```
pipeline.ipynb              trò + distill: train + dự đoán + submission.csv (chạy từ trên xuống)
pipeline_student.ipynb      trò tự học, không distill
requirements.txt
yolo_cfg/
  student_p2p4.yaml         trò, 191.145 tham số (bài nộp)
  teacher_p2p4.yaml         thầy, 2.580.073 tham số (chỉ để distill)
weights/
  yolo_kd_student.pt        trọng số trò đã nộp (checkpoint epoch 232, đã bỏ optimizer)
  yolo_teacher.pt           trọng số thầy (dùng khi muốn distill lại mà không train thầy)
  yolo_student.pt           trọng số trò tự học đã nộp (checkpoint epoch 290, đã bỏ optimizer)
  intent_textcnn.keras      TextCNN đã nộp
  intent_vocab.txt          vocab của TextCNN (dòng k = token id k; 0 = pad, 1 = từ lạ)
outputs/                    kết quả khi chạy (không đưa lên git)
```
