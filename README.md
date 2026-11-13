# Edge AI Challenge 2026 – Vòng loại

Detector vật thể tí hon (CenterNet + MobileNetV2) và phân loại intent (TextCNN) cho cuộc thi
[EDGE AI CHALLENGE 2026 LẦN 2 – Qualifiers](https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers).
Hạn nộp: **18/11/2026**.

## Tình trạng hiện tại

| | |
|---|---|
| Điểm public tốt nhất | **97,2** (run `A45-float`, dự đoán bằng model float, nộp 12/11/2026) |
| mAP50 val detection | 0,966 (float, run `s4`, chưa nộp) |
| Intent | accuracy 1,0 trên holdout, đã đạt tối đa |
| Mốc (cấu hình đang giữ) | `edgeai_float.py`: 320×240, ALPHA 0.45, STRIDE4, 196.307 tham số, 160 epoch |
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
| TTA (chạy model nhiều lần/ảnh, vd lật ngang) | **không được dùng** | 1 lần chạy/ảnh |
| Pretrained / dữ liệu ngoài | **chỉ được dùng ở vòng chung kết** | train từ đầu (`PRETRAINED = False`, `KD_WEIGHT = 0`) |
| Ensemble nhiều model | **không được dùng** | nộp 1 model |
| Distillation | được, nhưng thầy phải train từ đầu trên dữ liệu BTC | xem mục Kế hoạch |

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
| `TRAIN_ON_ALL` | `True` thì gộp cả val vào train. Chỉ dùng cho **lần chạy chốt cuối**, vì khi đó mAP val mất ý nghĩa |
| `QUICK_TEST` | `True` thì chạy 300 ảnh, 2 epoch để kiểm tra pipeline trong vài phút |

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
| A45-noaug | ALPHA 0.45 | float 0,9575 | giữ (mốc cũ) |
| A45-float | ALPHA 0.45, không tắt augment (Colab) | float 0,960, public 97,2 | giữ |
| kd | distill heatmap từ thầy YOLO11s (val 0,989–0,991) | 0,958 | không giữ (thấp hơn đối chứng A45-float 0,960) |
| **s4** | **thêm tầng stride 4 (lưới 80x60), 196.307 tham số** | **float 0,966** | **giữ, mốc hiện tại: cả 7 lớp tăng, toy_car +0,011** |
| pre35 | backbone ImageNet alpha 0.35 + stride 4, 138.611 tham số | float 0,967 | **không dùng ở vòng loại (cấm pretrained)**; ngang s4 với ít hơn 58k tham số -> để dành cho chung kết |
| (thử trên máy) | TTA lật ngang trên A45-noaug | 0,9645 | **không dùng: thể lệ cấm TTA** |

Kinh nghiệm rút ra:
- Độ phân giải đầu vào và độ rộng backbone (ALPHA) là hai đòn bẩy lớn nhất.
- Thay đổi augment, cách gán nhãn hay lịch học gần như không tác dụng.
- Tắt augment ở cuối không giúp gì: epoch tốt nhất của A45-noaug là 135, trước lúc tắt augment.
- Distill chỉ heatmap không giúp. AP@0.3 cao hơn AP@0.5 rõ ở toy_car/phone -> nút thắt là định vị box.

## Kế hoạch vòng loại (cập nhật 13/11/2026)

Cần thêm khoảng **+0,005 mAP** so với s4 (0,966 -> ~0,971) để lên 98 điểm.

### Luật chơi vòng loại (đã chốt, KHÔNG làm)

| Cấm ở vòng loại | Hệ quả cho repo |
|---|---|
| Pretrained / dữ liệu ngoài (chỉ được dùng ở chung kết) | `PRETRAINED = False`. Run `pre35` (0,967) và `kd` (thầy YOLO11s pretrained COCO) **không hợp lệ**, chỉ để tham khảo |
| TTA (chạy model nhiều lần mỗi ảnh) | 1 lần chạy/ảnh |
| Ensemble | nộp **1 model duy nhất**; không trộn dự đoán nhiều model, cũng không dùng ensemble làm thầy |
| Dùng ảnh test để gán nhãn / pseudo-label / train | thầy chỉ chạy trên ảnh train |

Distillation **được dùng**, nhưng thầy phải **train từ đầu trên dữ liệu BTC** (không mang kiến thức YOLO pretrained vào).
Thầy chỉ dùng lúc train, không có trong bài nộp; code thầy nộp kèm để tái tạo được.

Hướng đã loại: NECK_CH lớn hơn không còn chỗ (s4 đã 196.307 / 200.000 tham số); augment, gán nhãn, lịch học,
tắt augment cuối đều đã thử, không tác dụng.

### Vì sao run kd cũ không giúp, và sẽ làm khác thế nào

Run kd lấy **box cuối** của thầy rồi vẽ lại thành heatmap Gaussian -> gần như trùng nhãn thật, trò học thêm rất ít.
Nút thắt hiện tại là **định vị box** (AP@0.3 cao hơn AP@0.5 rõ ở toy_car, phone), mà kd cũ không distill box.

Cách mới (**distill dày, online**):
- Thầy (đóng băng) chạy **song song** trên đúng batch đã augment của trò -> không cần file nhãn mềm, không lệch augment.
- Loss thêm:
  - `KD_HM`: trò bắt chước **heatmap thô của thầy trên mọi ô** lưới (cả chỗ thầy phân vân giữa các lớp).
  - `KD_BOX`: L1 giữa box của trò và box của thầy, **có trọng số theo độ tự tin của thầy** (ô thầy chắc chắn có vật).
  - (chỉ khi thầy cùng họ CenterNet lớn hơn) `KD_FEAT`: bắt chước feature neck qua 1 lớp 1x1 chuyển kênh.
- Thầy và trò cùng kiến trúc CenterNet, cùng lưới 80x60 (stride 4) -> ghép từng ô trực tiếp, không phải giải mã box.

### Các bước và mọi trường hợp

Quy ước so sánh: **TB AP@0.5 của 7 lớp trên val** (seed 42, 20%) so với mốc **s4 = 0,9656**.
Chênh **< 0,003 = nhiễu**. Mỗi run khoảng 2,3 giờ GPU (T4 x2 Kaggle).

**Bước 1 – run `bakd`: tự distill (born-again), thầy = s4.**
> Trạng thái 13/11: **đang chạy trên Kaggle** (`kaggle_runs/bakd`, KD_HM 1.0, KD_BOX 0.5). Code distill online đã có trong
> `edgeai_float.py` (`KD_HM`, `KD_BOX`, `TEACHER_WEIGHTS`; mặc định 0 = tắt). Đã chạy thử trên máy: thầy nạp đúng,
> không bị cập nhật khi train, trò cập nhật đủ.
Trò = đúng kiến trúc s4, train lại từ đầu 160 epoch, thầy = trọng số `tiny_det_best.weights.h5` của kernel s4
(thêm `kernel_sources: phngnam296/edgeai-exp-s4`). Không tốn công train thầy.
- **≥ 0,969** (+0,003 trở lên): GIỮ, `bakd` thành mốc. Sang Bước 2 nếu còn ≥ 2 ngày, không thì sang Bước 3.
- **0,963 – 0,969** (nhiễu): chạy thêm **1 lần** đổi trọng số (`KD_BOX` x2, hoặc chỉ `KD_BOX` không `KD_HM`) để xem
  thành phần nào có tác dụng. Vẫn nhiễu -> bỏ born-again, sang Bước 2.
- **< 0,963** (tệ hơn): kiểm tra loss KD có lấn loss gốc không (log từng thành phần), giảm trọng số KD 1 lần;
  vẫn tệ -> sang Bước 2.
- Lỗi khi chạy (không nạp được thầy, lệch shape): sửa và chạy lại, không tính là kết quả.

**Bước 2 – thầy lớn train từ đầu (`tbig`) rồi distill (`tkd`).**
- `tbig`: cùng code, **bỏ trần tham số**: ALPHA 1.0, NECK_CH 128, STRIDE4, train trên 4.800 ảnh train
  (không val, không test). Ghi mAP val của thầy vào `results.csv`.
  - Thầy **≥ 0,975**: đủ mạnh -> chạy `tkd` (trò s4 + KD_HM + KD_BOX + KD_FEAT).
  - Thầy **< 0,975** (không hơn s4 rõ): distill khó giúp -> bỏ, sang Bước 3 với mốc tốt nhất hiện có.
  - Có thể chạy `tbig` trên Colab **song song** với `bakd` trên Kaggle để tiết kiệm thời gian.
- `tkd`: cùng tiêu chí giữ / nhiễu / tệ như Bước 1 (so với mốc hiện tại, có thể là `bakd`).

**Bước 3 – run chốt `final`: mốc tốt nhất + `TRAIN_ON_ALL = True`** (train trên cả 6.000 ảnh, dự kiến +0,003 đến +0,008).
- Không còn val để đo -> số epoch cố định bằng epoch tốt nhất của mốc (không chọn theo val).
- Nếu mốc có KD: dùng thầy đã train trên 4.800 ảnh (hoặc train lại thầy trên 6.000 ảnh nếu còn thời gian).
- Kiểm tra bắt buộc: đủ 4.000 ảnh test, ràng buộc OK, `submission.csv` hợp lệ.

**Bước 4 – nộp.**
- Nộp `final`. Nếu điểm public **thấp hơn** bản đang có (97,2 của A45-float) -> chọn lại bản cũ làm bài chính thức.
- Luôn giữ 1 bản dự phòng đã nộp được (`s4` hoặc `A45-float`).
- Không có gì cải thiện ở Bước 1–2 -> nộp **s4 + TRAIN_ON_ALL**.

### Lịch dự kiến

| Ngày | Việc |
|---|---|
| 13–14/11 | viết code distill online; chạy `bakd` (Kaggle) + `tbig` (Colab) song song |
| 15/11 | đọc kết quả; chạy run phụ của Bước 1 hoặc `tkd` |
| 16/11 | chạy `final` (TRAIN_ON_ALL) |
| 17/11 | nộp, kiểm tra điểm public, chọn bài chính thức |
| 18/11 | hạn chót – chỉ để dự phòng, không chạy thí nghiệm mới |

### Việc cho người thứ hai (làm song song với `bakd`)

**Việc 1 – chạy model thầy `tbig` (Bước 2), quan trọng nhất.**
File sẵn: `kaggle_runs/tbig/edgeai_kaggle.py` (= `edgeai_float.py` với ALPHA 0.75, NECK_CH 96, 474.579 tham số,
tắt intent). Vượt trần 200k là **cố ý** vì thầy không nộp; dòng `[LỖI] Số tham số` ở cuối log là bình thường.

Nên chạy bằng **tài khoản Kaggle của bạn** (30 giờ GPU/tuần riêng, không bị ngắt như Colab):
1. Vào trang cuộc thi, bấm Join (đồng ý luật) nếu chưa.
2. Code → New Notebook → File → Import / dán toàn bộ `kaggle_runs/tbig/edgeai_kaggle.py` vào 1 ô.
3. Add Input → Competition `egde-ai-challenge-2026-lan-2-qualifiers`.
4. Settings: Accelerator **GPU T4 x2**, Internet **tắt** (không cần).
5. **Save Version → Save & Run All (Commit)** rồi có thể tắt máy; chạy xong xem tab Output.
6. Xem log: dòng `epoch ...: ... | XXs/epoch`. Ước tính 2–2,5 lần s4 (~50 s/epoch) -> khoảng 4–5,5 giờ cho 160 epoch.

Colab cũng được (xem mục 2b), nhưng Colab miễn phí hay ngắt; `RESUME_WEIGHTS` chỉ nạp checkpoint tốt nhất,
**không train tiếp**, nên thầy có thể bị train thiếu.

Chạy xong, gửi lại:
- `tiny_det_float.weights.h5` (trọng số thầy) -> `git add -f kaggle_runs/tbig/output/tiny_det_float.weights.h5`
  rồi push (file bị `.gitignore` chặn nên phải `-f`), hoặc gửi trực tiếp.
- Log -> `kaggle_runs/tbig/output/` rồi `python kaggle_runs/_tools.py parse tbig "Thầy ALPHA 0.75, NECK_CH 96"`.
- Quyết định theo Bước 2: thầy **≥ 0,975** mới dùng để distill (`tkd`).

**Việc 2 – hỏi BTC cho chắc** (trên Discussion của cuộc thi hoặc kênh BTC):
- Distill từ model thầy **tự train từ đầu trên dữ liệu BTC** (thầy lớn hơn 200k tham số, không nộp) có hợp lệ ở vòng loại không?
- Câu "pretrained ... (chỉ ở VÒNG CHUNG KẾT)" là *chỉ được dùng* ở chung kết, hay *chỉ phải khai báo* ở chung kết?
- Câu trả lời ảnh hưởng trực tiếp tới `bakd`/`tkd` và `pre35`; ghi kết quả vào mục "Luật chơi vòng loại".

**Việc 3 – đọc kết quả và soát lại.**
- Khi `bakd` xong (log trong `kaggle_runs/bakd/output/`), so theo tiêu chí Bước 1 và ghi GIỮ / KHÔNG GIỮ vào `results.csv`.
- Đọc lại phần distill online trong `edgeai_float.py` (`make_kd_heat_loss`, `make_kd_box_loss`, đoạn `USE_KD`) xem có chỗ nào sai.

Quy ước chung: mọi run mới ghi vào `kaggle_runs/results.csv` kèm ghi chú GIỮ / KHÔNG GIỮ và lý do; pull trước khi sửa.

## Cần khai báo với BTC (vòng chung kết)

Vòng loại: bài nộp không dùng pretrained, dữ liệu ngoài hay ensemble. Danh sách dưới chỉ áp dụng nếu dùng lại các run `pre35` / `kd` ở chung kết.

- Trọng số pretrained MobileNetV2 (ImageNet, Keras Applications), dùng trong detector.
- Trọng số pretrained YOLO11n (COCO): chỉ dùng tham khảo, không có trong bài nộp.
- Trọng số pretrained YOLO11s (COCO, Ultralytics): khởi tạo model thầy (`teacher_yolo.py`), chỉ train trên
  tập train, sinh nhãn mềm `teacher_soft.json` để chưng cất khi train. Không có trong bài nộp.
  Hồ sơ nộp cần kèm `teacher_yolo.py` + `teacher_soft.json` (seed 42, 2 fold) để tái tạo được.
- Dữ liệu intent sinh thêm bằng cách thay tên vật thể trong câu train.
