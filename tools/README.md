# Giảm lệch miền camera OV7670

Ảnh dataset BTC là ảnh dựng 3D, sạch; ảnh OV7670 thật bị ám màu, mờ, nhiễu, tối góc và chia bậc màu RGB565.
Hai script ở đây giúp model quen với ảnh camera thật:

| Script | Làm gì |
|---|---|
| `merge_roboflow.py` | Gom dataset Roboflow (ảnh thật) về 7 class cuộc thi, ảnh 320x240, nhãn YOLO |
| `sim_ov7670.py` | Đo đặc điểm camera từ ảnh OV7670 thật, rồi biến ảnh bất kỳ thành ảnh "như OV7670 chụp" |

Chạy trong venv dự án, từ thư mục gốc repo (`F:\EdgeAI`).

## 1. Chụp ảnh camera thật (30-50 ảnh)

```powershell
python firmware\tools\client.py --host esp32s3-infer.local --out data\ov7670 capture --interval 2
```
(`--host`, `--out` phải đứng **trước** `capture`.)
Thay `esp32s3-infer.local` bằng IP của ESP nếu không vào được. Chụp nhiều cảnh, nhiều mức sáng (đèn, ánh sáng
cửa sổ, hơi tối), có và không có đồ vật. Ctrl+C để dừng.

## 2. Đo đặc điểm camera

```powershell
python tools\sim_ov7670.py fit --camera data\ov7670 --ref data\train\images --out tools\ov7670_profile.json
python tools\sim_ov7670.py preview --camera data\ov7670 --out outputs\sim_preview.png
```
Mở `outputs\sim_preview.png`: mỗi hàng là ảnh gốc | ảnh mô phỏng | ảnh camera thật. Cột giữa phải trông giống cột phải.

## 3. Gom dataset Roboflow

Trên Roboflow: Download Dataset -> Format **YOLOv8** (hoặc YOLO11) -> giải nén mỗi dataset vào `data\roboflow\<tên>`.

```powershell
python tools\merge_roboflow.py --src data\roboflow\cups data\roboflow\phones --out data\rf_merged
```
Đọc kỹ bảng ánh xạ class mà script in ra. Dòng ghi "(đoán)" cần kiểm tra; sửa bằng
`--map "Tên gốc=class,Tên khác=-"` (`-` = bỏ class đó).

Dataset sót nhãn (vd phone: điện thoại trắng không có box) -> bổ sung bằng model COCO **trước** khi gộp
(nhãn gốc sao lưu ở `labels_orig\`; xem preview trước bằng `--dry-run`):
```powershell
python tools\fill_labels.py --src data\Roboflow\phone_v1i_yolov8 --cls phone --model pipeline\weights\coco\yolo11x.pt --dry-run --preview outputs\fill_phone.jpg
python tools\fill_labels.py --src data\Roboflow\phone_v1i_yolov8 --cls phone --model pipeline\weights\coco\yolo11x.pt
```
COCO có cell phone, cup, bottle, book, mouse; không có box và xe đồ chơi.

**Bẫy nhãn thiếu:** dataset chỉ gán nhãn vài class thì các vật class khác trong ảnh bị coi là nền. Script cảnh báo
khi một nguồn thiếu class. Ưu tiên dataset gán đủ nhãn, hoặc dùng model thầy gán bổ sung.

## 3b. Thu nhỏ về tỉ lệ camera thật (zoom-out)

Camera OV7670 ở ~70cm: **~5 px/cm** (đo bằng tờ A4). Ảnh Roboflow chụp cận cảnh -> thu nhỏ theo kích thước thật
của từng loại vật (bảng `SIZE_CM` trong script) rồi ghép mosaic 2x2 thành ảnh 320x240:
```powershell
python tools\zoom_out.py --src data\rf_merged --out data\rf_zoom --px-per-cm 5.0 --preview outputs\zoom_preview.jpg
```
Đổi độ cao camera -> đo lại px/cm (cạnh dài A4 trên ảnh / 29.7) và chạy lại.

## 4. Sinh ảnh mô phỏng

```powershell
python tools\sim_ov7670.py apply --src data\rf_zoom --out data\rf_sim --copies 1 --keep-orig
python tools\sim_ov7670.py apply --src data\train --out data\train_sim --copies 1
```
Nhãn YOLO được chép nguyên (mô phỏng không dịch chuyển pixel).

## 5. Train và đo

**Tăng giới hạn box 96px -> 144px (0 tham số thêm):** chạy trước khi train/val/predict, trong mọi notebook/script:
```python
import sys; sys.path.insert(0, r"F:\EdgeAI\tools")
import yolo_reach; yolo_reach.apply((1.0, 1.0, 1.5))
```
Model train như vậy thì khi chạy trên ESP phải đặt `#define YOLO_DIST_SCALE_LAST 1.5f` trong
`firmware\main\logic.h` (model cũ: giữ `1.0f`), và `verify_det.py --dist-scale 1.5`.

Train trên `data\train` + `data\train_sim` + `data\rf_sim`. Luôn đo trên một tập ảnh **camera thật đã gán nhãn**
(không đưa vào train): đó mới là con số cho biết model chạy thật tốt hơn hay không.

Lưu ý thể lệ: vòng loại cấm dữ liệu ngoài; kiểm tra thể lệ vòng chung kết trước khi dùng cho bài nộp chính thức.
