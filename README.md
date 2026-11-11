# Edge AI Challenge 2026 – tuân thủ ràng buộc ESP32-S3 ngay từ Warm-up

## Cấu trúc thư mục

```
code/
  data/                      <- dữ liệu Kaggle: classes.txt, *.csv, train/, test/
  common.py                  đường dẫn, đọc nhãn, chia train/val cố định
  metrics.py                 mAP@0.5 (101 điểm, giống BTC) để tự chấm trên val
  intent_text.py             chuẩn hóa câu, tăng cường dữ liệu, tên vật thể
  det_tiny_data.py           nhãn CenterNet, augmentation, giải mã box (numpy)
  det_tiny_model.py          mô hình ~131k tham số + loss (TensorFlow)
  step1_explore.py           thống kê dữ liệu
  step2_intent_textcnn.py    TextCNN -> intent_predictions.csv + intent_int8.tflite
  step3_yolo_baseline.py     YOLO11n: CHỈ để biết "điểm trần", KHÔNG nộp
  step5_tiny_detector.py     detector tí hon -> det_predictions.json + tiny_det_int8.tflite
  step4_make_submission.py   ghép + kiểm tra -> outputs/submission.csv
```

## Thứ tự chạy

```bash
python step1_explore.py
python step2_intent_textcnn.py
python step5_tiny_detector.py     # CPU, ~5-15 phút
python step4_make_submission.py   # -> nộp outputs/submission.csv
python step3_yolo_baseline.py     # tuỳ chọn, GPU: so sánh mAP val với step5
```

## Ràng buộc đang tuân thủ (mục 7 thể lệ)

| Hạng mục | Yêu cầu | Detection | Intent |
|---|---|---|---|
| Đầu vào | ≤ 160×160 | 160×96 | 24 token |
| Tham số | ≤ 200.000 | ~131.000 | ~12.000 |
| Định dạng | TFLite INT8 | ✓ | ✓ |
| Dung lượng | ≤ 1,5 MB | ~200 KB | ~22 KB |
| Op | TFLite Micro | step5 tự kiểm tra | ✓ |

Bài nộp detection được dự đoán bằng chính file `.tflite` INT8, không phải bản float.

## Chuyển sang vòng loại
- Thay thư mục `data/` (classes.txt 7 lớp).
- `det_tiny_data.py`: đổi `INPUT_W, INPUT_H` cho ảnh 320×240 (4:3), ví dụ 160×128 hoặc 128×96.
- `intent_text.py`: thay `OBJECT_NAMES` bằng tên 7 đồ vật.

## Cần khai báo với BTC
- Trọng số pretrained MobileNetV2 (ImageNet, Keras Applications), dùng trong detector.
- Trọng số pretrained YOLO11n (COCO), chỉ dùng tham khảo, không có trong bài nộp.
- Dữ liệu intent sinh thêm bằng cách thay tên vật thể trong câu train.
