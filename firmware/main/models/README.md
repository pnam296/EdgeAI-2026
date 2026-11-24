# Trọng số model (mảng C `.cc` + `.h`)

Repo đã kèm trọng số baseline. Thay model của đội: cần đủ 3 cặp file: `det_model.cc/.h`, `textcnn_model.cc/.h`, `vocab.cc/.h`.

`baseline.ipynb` tự sinh ra ở 2 ô export (int8), copy vào đây rồi `idf.py build flash`:
- Phần 1 Detection → `model_int8.tflite` + `det_model.cc/.h`
- Phần 2 TextCNN → `textcnn_int8.tflite` + `textcnn_model.cc/.h` + `vocab.cc/.h` (luôn copy cùng nhau)

(Đặt `cc_dir` trong ô export để ghi thẳng vào thư mục này.)

Yêu cầu khớp với firmware:
- Detection: input int8 `1x240x320x3` (RGB/255), output NHWC `1x15x20x12` = obj, tx, ty, w, h + 7 class
  theo thứ tự `bottle cup phone book mouse box toy_car`.
- TextCNN (export qua Keras, không qua onnx2tf vì onnx2tf chèn op TFLite Micro không có và nối sai trục khi seq = emb = 32): input 32 token **int32** (token id; TFLite Micro không hỗ trợ int64 cho Gather), trọng số + output int8,
  output 5 intent `COUNT EXIST LIST LOCATION OUT_OF_SCOPE`.

Arena, ngưỡng conf/NMS, danh sách op: `main/model_config.h`.
Đổi class/intent: sửa `CLASS_NAMES`, `INTENT_NAMES`, `KW`, `VN` trong `main/logic.c` và `NAMES` trong `main/index.html`.
