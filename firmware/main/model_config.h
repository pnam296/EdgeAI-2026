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

/* Nguong mac dinh. Chinh bang tools/tune_answer.py: tim theo DO CHINH XAC CAU TRA LOI cua luat chung ket
 * (COUNT+EXIST 35%, LIST 25%, LOCATION 30%), khong theo mAP; script in san doan #define de dan vao day.
 * Web/API doi tung lan chay: ?conf=0.5 (ghi de nguong MOI lop) &loc=0.1 &iou=0.6 &agn=1.
 *  CLASS_CONF_INIT: nguong tung lop {bottle, cup, phone, book, mouse, box, toy_car} cho COUNT/EXIST/LIST
 *  LOC_FLOOR      : LOCATION hoi lop X ma khong co box X du nguong -> lay box X diem cao nhat >= LOC_FLOOR
 * Gia tri duoi: model cam3R3B (kaggle_runs/cam3R3B: thay cam3 + tro P2P4 khoi tao tu cam3, du lieu gen5, KD 2.0,
 * KHONG co anh camera that; P4 x1.5, INT8). Do dung cau tra loi voi cau hinh duoi (anh chua thay): khan200 0.614,
 * khan875 0.601, khan1075 0.603, 9 anh BTC 0.531, real55 0.434. Model truoc: cam3A (4153b48). */
#define CONF_THRESH      0.45f
/* NMS 0.45, GOP LOP (nguoi dung chon: tranh 2 label khac lop de len cung 1 vat).
 * cam3R2B + nguong duoi: khan200 0.621, khan875 0.595, BTC mau 0.519 (gop lop 0.30: 0.617/0.596/0.519;
 * 0.30 theo lop + chi bottle 0.35: 0.619/0.601/0.537). Truoc do: 0.30 theo lop (c6fe430). */
#define NMS_IOU          0.45f
#define NMS_AGNOSTIC     1        /* 1: bo box lop khac trung len cung 1 vat; 0: chi NMS trong cung lop */
/* bottle 0.30: chai nhua TRONG SUOT tren nen sang diem thap (0.45 chi bat 62% chai that, 0.30 ~70%);
 * dung cau tra loi khan1075 0.576 -> 0.580, BTC val 0.886 -> 0.888 (tune_answer --eval-cls bottle=0.3)
 * Thu cam3R2B: bottle 0.35 (nguoi dung chot): khan200 0.619, khan875 0.601, BTC mau 0.537 (0.30: 0.618/0.600/0.552; 0.40 cho khan200 0.618 -> 0.618, khan875 0.600 -> 0.602, BTC mau 0.552 -> 0.530) */
/* cup 0.65, mouse 0.70 (nguoi dung chon): bot bao nham coc/chuot */
#define CLASS_CONF_INIT  {0.35f, 0.65f, CONF_THRESH, CONF_THRESH, 0.70f, CONF_THRESH, CONF_THRESH}
#define LOC_FLOOR        0.100f   /* chung ket co DU do tren ban -> cau vi tri hau het hoi vat CO that: du phong co loi.
                                    tune_answer (10% hoi vat vang mat): khan200 LOC 0.789 (vat co 0.790, vang 0.783),
                                    BTC val 0.971. 0.08 hay "bia" vi tri khi vat vang mat; neu de hoi nhieu vat vang -> 0.25 */

/* YOLO: so bin DFL moi canh (reg_max trong file .yaml trong yolo_cfg/). Grid baseline khong dung. */
#define YOLO_REG_MAX     4
/* He so tam voi muc P4 (model train voi tools/yolo_reach.py): dat YOLO_DIST_SCALE_LAST trong main/logic.h */

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
