# EDGE AI CHALLENGE 2026 (LẦN 2) - VISUAL QUESTION ANSWERING

## 1. Mục tiêu & Lịch trình cuộc thi
- **Vòng Loại (09/11 - 18/11/2026 - Kaggle Online):** Tối ưu điểm số trên Private Leaderboard với 2 bài toán độc lập:
  - **Object Detection (70% điểm):** Đánh giá bằng `mAP@0.5` trên 7 lớp (`0: bottle`, `1: cup`, `2: phone`, `3: book`, `4: mouse`, `5: box`, `6: toy car`). Ảnh gốc kích thước `320x240`, dữ liệu synthetic (6.000 train / 4.000 test).
  - **Intent Classification (30% điểm):** Đánh giá bằng `Accuracy` trên 5 nhóm ý định tiếng Việt (`COUNT`, `EXIST`, `LIST`, `LOCATION`, `OUT_OF_SCOPE`). Dữ liệu cân bằng (6.000 train / 4.000 test, mỗi lớp 800 mẫu test). Lưu ý: ~43% câu hỏi test dùng cấu trúc diễn đạt mới chưa từng xuất hiện trong tập train.
- **Vòng Chung Kết (29/11/2026):** Triển khai toàn bộ hệ thống (Detection + Intent + Logic VQA) trực tiếp trên vi điều khiển **ESP32-S3 N16R8** + camera **OV7670** (không dùng server/cloud).

## 2. QUY ĐỊNH BẮT BUỘC (TUYỆT ĐỐI KHÔNG VI PHẠM)
1. **Cấm Pretrained & Dữ liệu ngoài ở Vòng Loại (Điều 6):**
   - Trong suốt Vòng loại, **TUYỆT ĐỐI KHÔNG** dùng trọng số pretrained bên ngoài (như ImageNet `weights="imagenet"`, YOLO pretrained) hay dữ liệu ngoài. Mọi mô hình (kể cả mô hình thầy `teacher_yolo.py` nếu dùng để Knowledge Distillation) **BẮT BUỘC phải train từ đầu (from scratch)** chỉ trên 6.000 ảnh BTC cung cấp.
   - Chỉ được phép dùng pretrained/dữ liệu ngoài ở Vòng Chung kết (và phải khai báo).
2. **Cấm rò rỉ dữ liệu Test (Điều 4):** Không dùng tập Test để gán nhãn giả (pseudo-label), tinh chỉnh tham số hay huấn luyện dưới mọi hình thức.
3. **Cấm TTA (Test-Time Augmentation):** Chỉ suy luận 1 lần duy nhất cho mỗi ảnh test.
4. **Ràng buộc kiến trúc & phần cứng:**
   - Kích thước ảnh đầu vào Detector: `<= 320x240` (chia hết cho 16).
   - Số tham số Detector: `<= 200,000` params.
   - Dung lượng file trọng số/mô hình: `<= 1.5 MB` (1,536 KB). Vòng loại được phép nộp kết quả từ mô hình Float32; Vòng chung kết bắt buộc lượng tử hóa INT8 tương thích các toán tử của **TFLite Micro (`TFLM_OPS`)**.
5. **Khả năng tái tạo (Điều 7 & 8):** Mọi thử nghiệm phải cố định `SEED = 42` và lưu đầy đủ checkpoint, cấu hình, script để đối soát.

## 3. Cấu trúc mã nguồn Repository
- **Module lõi:**
  - `common.py`: Đọc dữ liệu, tự động nhận diện định dạng nhãn (`yolo`, `xyxy_px`, `xyxy_norm`), chia tập train/val (80/20).
  - `metrics.py`: Tính toán chuẩn xác `mAP@0.5` và `AP@0.3` theo từng lớp để chẩn đoán lỗi (box lệch vs nhầm lớp).
  - `det_tiny_data.py`: Pipeline nạp ảnh, cache kích thước `640`, augment ảnh + bounding box, encode Gaussian heatmap và nhãn mềm của teacher.
  - `det_tiny_model.py`: Kiến trúc Tiny CenterNet (Backbone MobileNetV2 tùy chỉnh `ALPHA` + Neck FPN tích hợp nhánh `STRIDE4` lưới `80x60` để bắt vật nhỏ như `toy car` + lớp `ResizeNearest` tương thích TFLM), Focal Loss & Box Loss.
  - `intent_text.py` / `step2_intent_textcnn.py`: Tiền xử lý tiếng Việt không dấu/chuẩn hóa, từ điển động, tăng cường dữ liệu bằng từ điển (`DICT_AUGMENT` tráo danh từ trong/ngoài 7 lớp) và mô hình TextCNN (Conv1D kernel 2, 3, 4).
- **Pipeline thực thi & Thử nghiệm:**
  - `teacher_yolo.py`: Huấn luyện mô hình thầy (YOLO) **từ đầu (scratch)** để xuất `teacher_soft.json` phục vụ Knowledge Distillation (`KD_WEIGHT`).
  - `edgeai_float.py` / `step5_tiny_detector.py` / `predict_float.py`: Huấn luyện, đánh giá và suy luận mô hình Detector Float32 cho vòng loại.
  - `edgeai_kaggle.py` / `kaggle_train_all.py`: Script tổng hợp chạy end-to-end trên Kaggle GPU (T4 x2) và xuất `submission.csv` (`task_type`, `sample_id`, `prediction`).
  - `kaggle_runs/`: Thư mục lưu log, cấu hình và kết quả các lần chạy thực nghiệm.

## 4. Nguyên tắc hỗ trợ của AI Assistant (Gemini CLI)
- **Kiểm tra tuân thủ trước tiên:** Khi đề xuất chỉnh sửa code hoặc cấu hình, luôn kiểm tra xem có vi phạm quy định Vòng loại (đặc biệt là `PRETRAINED = False`, giới hạn `< 200k` params, `< 1.5MB`, và không dùng pretrained weights cho `teacher_yolo.py`) hay không.
- **Đồng bộ hóa module:** Dự án có cả cấu trúc modular (`det_tiny_model.py`, `det_tiny_data.py`...) và bản gộp chạy Kaggle (`edgeai_kaggle.py`). Khi sửa logic ở module lẻ, luôn nhắc hoặc cập nhật đồng bộ sang file chạy Kaggle tương ứng.
- **Tối ưu có cơ sở:** Mọi đề xuất cải thiện `mAP@0.5` (ví dụ: chỉnh `ALPHA`, `NO_AUG_LAST`, `KD_WEIGHT`, `TRAIN_ON_ALL` ở lần chốt cuối) hoặc `Intent Accuracy` phải dựa trên phân tích log trong `kaggle_runs/` và đặc điểm dữ liệu cuộc thi.
- **Ngôn ngữ:** Trả lời bằng tiếng Việt ngắn gọn, đi thẳng vào kỹ thuật và code.