# Edge AI Challenge 2026 – Vòng loại

Detector vật thể tí hon (CenterNet + MobileNetV2) và phân loại intent (TextCNN) cho cuộc thi
[EDGE AI CHALLENGE 2026 LẦN 2 – Qualifiers](https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers).
Hạn nộp: **18/11/2026**.

## Tình trạng hiện tại

| | |
|---|---|
| Điểm public tốt nhất | **96,88** (run `A45-noaug`, dự đoán bằng model float) |
| mAP50 val detection | 0,9575 (float), 0,9645 nếu dùng TTA lật ngang (chưa nộp, chờ xác nhận thể lệ) |
| Intent | accuracy 1,0 trên holdout, đã đạt tối đa |
| Mốc (cấu hình đang giữ) | `edgeai_float.py`: 320×240, ALPHA 0.45, 184.723 tham số, 160 epoch |
| Mục tiêu | 98 điểm, tức mAP50 detection khoảng 0,971 trên tập public |

**Công thức điểm** (đã khớp với điểm public): `điểm = 0,7 × mAP50 detection × 100 + 30 × accuracy intent`.
mAP trên tập public thường thấp hơn mAP val khoảng 0,002.

## Ràng buộc vòng loại

| Hạng mục | Giới hạn | Mốc hiện tại |
|---|---|---|
| Đầu vào | ≤ 320×240 | 320×240 |
| Số tham số detector | ≤ 200.000 | 184.723 |
| Dung lượng model | ≤ 1,5 MB | khoảng 1.090 KB (trọng số float32) |
| Lượng tử hóa INT8 | **không bắt buộc ở vòng loại** (thông báo BTC 11/11/2026) | dùng model float |

> Từ 11/11/2026, script mới **không** xuất TFLite hay INT8 nữa. Các run cũ trong `kaggle_runs/` và file
> `edgeai_kaggle.py` vẫn còn phần INT8, chỉ giữ lại làm lịch sử.

## Cấu trúc thư mục

```
edgeai_float.py            ★ SCRIPT MỐC: train detector + intent + tạo submission.csv (float, 1 file)
predict_float.py           dự đoán lại bằng model .keras đã train -> submission_float.csv (không cần GPU)
requirements.txt           thư viện Python
kaggle_runs/
  results.csv              ★ bảng kết quả MỌI thí nghiệm (ghi thêm sau mỗi run)
  state.json               danh sách thay đổi đã được giữ vào mốc
  _tools.py                đọc log Kaggle -> ghi results.csv; đánh dấu run được giữ
  _wait.ps1                chờ một kernel Kaggle chạy xong
  <mã run>/                mỗi thí nghiệm 1 thư mục
    edgeai_kaggle.py       script đúng như lúc chạy
    kernel-metadata.json   cấu hình kernel Kaggle (id, GPU, dữ liệu cuộc thi)
    output/*.log           log Kaggle (model và submission không đưa lên git)
  A45-float/               bản mốc đã chỉnh để chạy trên Colab (Drive + chạy tiếp khi bị ngắt)

edgeai_kaggle.py           script cũ (có INT8), là gốc của _tools.py prepare
step1..step5_*.py, common.py, metrics.py, det_tiny_*.py, intent_text.py
                           code chia nhỏ từ giai đoạn Warm-up, để tham khảo
data/                      dữ liệu cuộc thi (KHÔNG có trên git, mỗi người tự tải)
```

## Cài đặt

```bash
git clone https://github.com/pnam296/EdgeAI-2026.git
cd EdgeAI-2026
python -m venv .
Scripts\activate                 # Windows;  macOS/Linux: source bin/activate
pip install -r requirements.txt
```

**Dữ liệu** (phải tham gia cuộc thi trên Kaggle trước):

```bash
# cần file kaggle.json: Kaggle -> Settings -> API -> Create New Token
kaggle competitions download -c egde-ai-challenge-2026-lan-2-qualifiers
# giải nén vào data/  -> data/classes.txt, data/train/, data/test/, data/*.csv
```

**Không commit** `data/`, model (`.keras`, `.h5`, `.tflite`, `.pt`), submission hay `kaggle.json`.
Các file này đã có trong `.gitignore`.

## Chạy một thí nghiệm

Một run gồm train detector (khoảng 1 giờ 45 phút trên GPU T4), train intent (vài phút) và tạo `submission.csv`.
Trên máy không có GPU thì chỉ nên chạy thử (`QUICK_TEST = True`).

### 1. Tạo thư mục run

```bash
mkdir kaggle_runs/<mã>                    # vd: kaggle_runs/neck80
cp edgeai_float.py kaggle_runs/<mã>/edgeai_kaggle.py
```

Sửa **đúng 1 thay đổi** trong phần `CẤU HÌNH` ở đầu file (vd `NECK_CH = 64` → `NECK_CH = 80`), để biết
chắc thay đổi nào tạo ra kết quả. Các tùy chọn chính:

| Biến | Ý nghĩa |
|---|---|
| `INPUT_W, INPUT_H` | kích thước đầu vào (≤ 320×240, chia hết cho 16) |
| `ALPHA`, `NECK_CH` | độ rộng backbone MobileNetV2 / cổ mạng, quyết định số tham số (≤ 200k) |
| `EPOCHS`, `BATCH`, `LR` | lịch train (cosine theo `EPOCHS`) |
| `PATIENCE` | > 0 thì bật dừng sớm thay cho cosine |
| `TTA_FLIP` | `True` thì dự đoán test bằng TTA lật ngang. **Chỉ bật khi thể lệ cho phép** chạy model 2 lần mỗi ảnh |
| `TRAIN_ON_ALL` | `True` thì gộp cả val vào train. Chỉ dùng cho **lần chạy chốt cuối**, vì khi đó mAP val mất ý nghĩa |
| `QUICK_TEST` | `True` thì chạy 300 ảnh, 2 epoch để kiểm tra pipeline trong vài phút |

Script luôn in ra mAP val khi dùng TTA để tiện so sánh, kể cả khi `TTA_FLIP = False`.

### 2a. Chạy trên Kaggle (CLI)

Sửa `kaggle_runs/<mã>/kernel-metadata.json`, chép từ một run có sẵn rồi đổi các trường:
- `"id": "<username-kaggle-của-bạn>/edgeai-exp-<mã>"` và `"title": "edgeai exp <mã>"`
- giữ `"enable_gpu": true`, `"machine_shape": "NvidiaTeslaT4"`, `"competition_sources": ["egde-ai-challenge-2026-lan-2-qualifiers"]`

```bash
kaggle kernels push -p kaggle_runs/<mã>
kaggle kernels status <username>/edgeai-exp-<mã>
# xong rồi thì tải log + kết quả về:
kaggle kernels output <username>/edgeai-exp-<mã> -p kaggle_runs/<mã>/output
```

Quota GPU Kaggle là **30 giờ mỗi tuần cho mỗi tài khoản**, mỗi run tốn khoảng 2 giờ. Cột `gio_gpu` trong
`results.csv` ghi số giờ đã dùng, `python kaggle_runs/_tools.py gpu_hours` in ra tổng.

### 2b. Chạy trên Colab

Dùng `kaggle_runs/A45-float/edgeai_kaggle.py` làm gốc. File này đã hỗ trợ Drive.

1. Runtime → Change runtime type → **T4 GPU**
2. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
3. Chép dữ liệu vào `/content`, vì đọc ảnh trực tiếp từ Drive rất chậm: `!unzip -q /content/drive/MyDrive/data.zip -d /content/data`
4. Dán toàn bộ script vào một ô code rồi chạy. Kết quả và checkpoint được ghi vào `MyDrive/edgeai_outputs/<tên>/` (sửa biến `DRIVE_OUT`).
5. Nếu Colab bị ngắt giữa chừng: đặt `RESUME_WEIGHTS = ".../tiny_det_best.weights.h5"` rồi chạy lại. Script sẽ bỏ qua train và làm tiếp phần dự đoán và submission.

Chạy xong, chép log (nội dung ô output) vào `kaggle_runs/<mã>/output/run.log` để lưu lại.

### 3. Ghi kết quả

```bash
# log Kaggle trong kaggle_runs/<mã>/output/*.log -> thêm 1 dòng vào results.csv
python kaggle_runs/_tools.py parse <mã> "<mô tả thay đổi>"
# nếu run tốt hơn mốc thì đánh dấu giữ, rồi cập nhật edgeai_float.py theo thay đổi đó
python kaggle_runs/_tools.py accept <mã>
```

Run chạy tay (Colab) mà `parse` không đọc được thì ghi bằng
`python kaggle_runs/_tools.py manual <mã> "<mô tả>" map_float=0.957 ap_bottle=0.965 ...`.
Luôn ghi kết luận vào cột `ghi_chu`: **GIỮ** hoặc **KHÔNG GIỮ**, kèm lý do.

**Quy tắc giữ:** so mAP val của run với mốc. Chênh lệch dưới khoảng ±0,003 coi như nhiễu, không giữ,
trừ khi lớp yếu nhất (toy_car, mouse) tăng rõ.

### 4. Nộp bài

```bash
kaggle competitions submit egde-ai-challenge-2026-lan-2-qualifiers \
  -f kaggle_runs/<mã>/output/submission.csv -m "<mã>: <mô tả>, val mAP50 0.xxx"
kaggle competitions submissions egde-ai-challenge-2026-lan-2-qualifiers   # xem điểm
```

Mỗi ngày chỉ được nộp vài lần, nên **chỉ nộp khi mAP val vượt mốc**. Báo cho nhau trước khi nộp.

Dự đoán lại bằng một model `.keras` có sẵn (vd thử TTA, đổi cách decode) mà không cần train:

```bash
python predict_float.py kaggle_runs/<mã>     # cần kaggle_runs/<mã>/output/tiny_det.keras + submission.csv
```

## Làm việc chung bằng git

```bash
git pull                                   # TRƯỚC khi bắt đầu làm
# ... làm 1 thí nghiệm trong kaggle_runs/<mã>/ ...
git add -A
git commit -m "run <mã>: <thay đổi> -> val mAP 0.xxx (GIỮ/KHÔNG GIỮ)"
git push
```

- **Mỗi người một thư mục run** và báo trước mình đang chạy thí nghiệm gì, để không tốn GPU chạy trùng.
- `results.csv` và `edgeai_float.py` là file chung: `git pull` ngay trước khi sửa, sửa xong push luôn.
- Nếu `git pull` báo xung đột ở `results.csv`: giữ **cả hai** dòng mới.

## Nhật ký thí nghiệm (tóm tắt, chi tiết trong `kaggle_runs/results.csv`)

| Run | Thay đổi | mAP val | Kết luận |
|---|---|---|---|
| 1b | đầu vào 160×128 → 320×240 | 0,918 (INT8 0,909) | giữ |
| 1b-a35 | ALPHA 0.25 → 0.35 | 0,939 | giữ |
| 3-a35-160 | 100 → 160 epoch | 0,954 (INT8 0,948) | giữ |
| 2b | NECK_CH 64 → 96 | +0,001 | không giữ (nhiễu) |
| 5 | augment scale 0.55 → 0.4 | −0,011 | không giữ |
| 8 | vùng gán nhãn quanh tâm 0.54 → 0.4 | −0,001 | không giữ |
| es | dừng sớm thay cosine | float 0,957, INT8 0,945 | không giữ (khi còn bắt buộc INT8) |
| noaug | tắt augment 15 epoch cuối | 0,952 | không giữ (nhiễu) |
| A45 | ALPHA 0.45 + EMA/GIoU/WH_SQRT/SPLIT_BOX | float 0,956, INT8 0,941 | không giữ |
| **A45-noaug** | **ALPHA 0.45** | **float 0,9575** | **giữ, mốc hiện tại** |
| (thử trên máy) | TTA lật ngang trên A45-noaug | 0,9645 | chờ xác nhận thể lệ |

Kinh nghiệm rút ra:
- Độ phân giải đầu vào và độ rộng backbone (ALPHA) là hai đòn bẩy lớn nhất.
- Thay đổi augment, cách gán nhãn hay lịch học gần như không tác dụng.
- Tắt augment ở cuối không giúp gì: epoch tốt nhất của A45-noaug là 135, trước lúc tắt augment.

## Hướng tiếp theo để lên 98

1. **TTA lật ngang**: +0,007 trên val, không cần train. Cần xác nhận thể lệ.
2. **Dùng hết ngân sách tham số**: NECK_CH 64 → 80 (khoảng 199k), dự kiến +0,001 đến +0,003.
3. **Chưng cất từ model thầy**: train YOLO11s không giới hạn tham số, dùng dự đoán của nó làm nhãn mềm. Dự kiến +0,005 đến +0,015.
4. **TRAIN_ON_ALL** ở lần chạy chốt cuối: +0,003 đến +0,008.

## Cần khai báo với BTC

- Trọng số pretrained MobileNetV2 (ImageNet, Keras Applications), dùng trong detector.
- Trọng số pretrained YOLO11n (COCO): chỉ dùng tham khảo, không có trong bài nộp.
- Dữ liệu intent sinh thêm bằng cách thay tên vật thể trong câu train.
