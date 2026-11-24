#pragma once
/* Cau hinh model. Doi ten mang theo file .h trong main/models/ */
#include "det_model.h"
#include "textcnn_model.h"
#include "vocab.h"

#define DET_MODEL_DATA   det_model        /* mang trong det_model.cc */
#define TXT_MODEL_DATA   textcnn_model    /* mang trong textcnn_model.cc */
#define VOCAB_DATA       vocab            /* vocab.cc: vocab[k] co token id k+2 */
#define VOCAB_LEN        vocab_len

/* Arena PSRAM. Log khoi dong in "arena dung X / Y KB": giam Y xuong ~X + 10% neu can tiet kiem.
 * YOLO P2P4 320x240 int8 uoc tinh ~1.5-2 MB (nhieu feature map 80x60 song cung luc). */
#define DET_ARENA_KB     3072
#define TXT_ARENA_KB     128

/* Nguong mac dinh (web/API co the doi tung lan chay: ?conf=0.5&iou=0.6&agn=1).
 * tools/verify_det.py in bang precision/recall theo conf: tren val, conf 0.5-0.6 cho precision 0.96-0.98
 * va ti le anh dem dung cao nhat. Anh camera that thuong co score thap hon -> chinh lai sau khi thu tren board. */
#define CONF_THRESH      0.50f
#define NMS_IOU          0.60f
#define NMS_AGNOSTIC     1        /* 1: bo box lop khac trung len cung 1 vat (giam false positive) */

/* YOLO: so bin DFL moi canh (reg_max trong file .yaml trong yolo_cfg/). Grid baseline khong dung. */
#define YOLO_REG_MAX     4

/* Danh sach op: cac script trong tools/ tu sinh models/ops_gen.h tu cac model .cc dang co.
 * Khong co ops_gen.h (repo goc) -> dung danh sach cua baseline. */
#if __has_include("ops_gen.h")
#include "ops_gen.h"
#else
#define RESOLVER_SIZE    12
#define REGISTER_OPS(r) do { \
    (r).AddConv2D(); (r).AddMaxPool2D();                          /* detection */          \
    (r).AddLess(); (r).AddCast(); (r).AddMul(); (r).AddAdd();     /* textcnn ban onnx2tf */ \
    (r).AddGather(); (r).AddTranspose(); (r).AddPad(); (r).AddReshape();                   \
    (r).AddReduceMax(); (r).AddFullyConnected();                                           \
} while (0)
#endif
