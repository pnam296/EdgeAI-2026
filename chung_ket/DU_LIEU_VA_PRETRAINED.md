# Dữ liệu và mô hình pretrained đã sử dụng

## A. Dữ liệu đội tự thu thập / tự tạo

| Dữ liệu | Ở đâu |
|---|---|
| **Ảnh camera thật OV7670**: 1.222 ảnh 320×240, tự chụp trên nền trắng, nhãn YOLO tự gán và sửa tay. `val_10phantram.txt` là 125 ảnh giữ lại để kiểm tra (không train) | **Trong repo:** [`data/cam_real_ov7670/`](data/cam_real_ov7670) · Kaggle: https://www.kaggle.com/datasets/phngnam296/edgeai-cam-real2 |
| Dữ liệu train detect đã gộp (BTC + ảnh mô phỏng giống BTC + Roboflow đã đổi về 7 lớp) | https://www.kaggle.com/datasets/phngnam296/edgeai-cam-data · https://www.kaggle.com/datasets/phngnam296/edgeai-cam-real · https://www.kaggle.com/datasets/phngnam296/edgeai-cam3-data · https://www.kaggle.com/datasets/dohuuminhtri/edgeai-cam3-all · https://www.kaggle.com/datasets/dohuuminhtri/edgeai-hotwheels |
| Script sinh dữ liệu, yaml, công cụ | Trong repo: [`../tools/`](../tools), [`../yolo_cfg/`](../yolo_cfg) · Kaggle: https://www.kaggle.com/datasets/phngnam296/edgeai-tools3 |
| Ảnh tổng hợp (dán vật lên nền giấy A0 trắng) gen4 / gen5, kho vật cắt từ Open Images / TACO | Sinh lại bằng notebook [`notebooks/01…04`](notebooks); bản đã chạy: https://www.kaggle.com/code/dohuuminhtri/edgeai-oicrops · https://www.kaggle.com/code/dohuuminhtri/edgeai-tacocrops · https://www.kaggle.com/code/dohuuminhtri/edgeai-gen4 · https://www.kaggle.com/code/dohuuminhtri/edgeai-gen5 |
| Câu hỏi intent sinh thêm (thay từ theo từ điển, chỉ lúc train) | Sinh trong [`../tools/train_intent.py`](../tools/train_intent.py) từ `intent_train.csv` của BTC |

## B. Dữ liệu từ bên ngoài (link gốc)

| Dữ liệu | Link |
|---|---|
| Dữ liệu BTC (ảnh, nhãn, `intent_train.csv`, `intent_test.csv`) | https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers (không đưa lại lên GitHub theo thể lệ) |
| COCO 2017 | https://cocodataset.org · https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset |
| LVIS (nhãn trên ảnh COCO) | https://www.lvisdataset.org |
| Open Images V7 | https://storage.googleapis.com/openimages/web/index.html |
| TACO | http://tacodataset.org · https://www.kaggle.com/datasets/kneroma/tacotrashdataset |

Roboflow Universe:

| Lớp | Link |
|---|---|
| bottle | https://universe.roboflow.com/-2gabz/bottle-5eahy · https://universe.roboflow.com/new-workspace-s4lik/bottle-pk64x |
| cup | https://universe.roboflow.com/noe-ew9vt/cup-detection-cevbw · https://universe.roboflow.com/test-oyysq/cup-detection-ne45p |
| book | https://universe.roboflow.com/akm-sazur/book-detection-rjt0m · https://universe.roboflow.com/light-a0cik/light-book-tqkoa · https://universe.roboflow.com/testyolo-jjozu/book-t29gb |
| mouse | https://universe.roboflow.com/pnu-og7cn/mouse-vfvw6 |
| box | https://universe.roboflow.com/are-bla-gpxot/cardboard-box-u35qd · https://universe.roboflow.com/hesi/box-39rbe · https://universe.roboflow.com/project-wfbsj/box-plzjm |
| toy_car | https://universe.roboflow.com/cardetection-czxod/car_toy · https://universe.roboflow.com/sample-msd01/cars-llo2h · https://universe.roboflow.com/group-project-konh2/hot-wheels-recognition |

## C. Mô hình pretrained

| Model | Dùng để | Weight |
|---|---|---|
| YOLO11s (Ultralytics, COCO) | khởi tạo backbone mô hình **thầy** (notebook 05). Không nạp lên ESP | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt |
| YOLO11x (Ultralytics, COCO) | gán nhãn bổ sung khi tạo dữ liệu (notebook 03, 06) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt |
| YOLO-World v2 L (Ultralytics) | hỗ trợ gán nhãn trước ảnh camera thật (`tools/smart_prelabel.py`), sau đó sửa tay | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt |
| OpenCLIP ViT-B-32 `laion2b_s34b_b79k` | lọc ảnh sách khi cắt vật Open Images (notebook 01) | https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K |
| fastText tiếng Việt, rút gọn 32 chiều | khởi tạo embedding TextCNN intent (notebook 10) | trong repo: [`../pipeline/weights/vi_fasttext_32d.npz`](../pipeline/weights/vi_fasttext_32d.npz) · gốc: https://fasttext.cc/docs/en/crawl-vectors.html |

**Mô hình chạy trên ESP32-S3** không dùng trọng số pretrained bên ngoài:
- Detect: trò YOLO P2P4 191.145 tham số, kiến trúc tự thiết kế ([`../yolo_cfg/student_p2p4.yaml`](../yolo_cfg/student_p2p4.yaml)), train từ đầu rồi distill từ thầy.
- Intent: TextCNN 186.117 tham số.

Weight thi đấu nằm trong [`weights/`](weights); bản đã xuất `.cc` nằm trong [`../firmware/main/models/`](../firmware/main/models).
