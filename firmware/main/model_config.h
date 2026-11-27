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
 * Gia tri duoi: model cam3A (yolo_cam3A_last.pt: cam1 + COCO/LVIS dan theo mask + Roboflow da sua + Hot Wheels,
 * KHONG co anh camera that; P4 x1.5, INT8). Do dung cau tra loi (anh chua thay): 200 anh khan trang 0.662,
 * 1075 anh khan trang 0.606 (cam2: 0.469), 9 anh BTC 0.404 (cam2: 0.347). mAP50 INT8 khan200 0.641 (float 0.672).
 * Nguong giu nhu cam2 (chua chinh lai cho cam3A). */
#define CONF_THRESH      0.45f
#define NMS_IOU          0.40f
#define NMS_AGNOSTIC     1        /* 1: bo box lop khac trung len cung 1 vat (giam false positive) */
#define CLASS_CONF_INIT  {CONF_THRESH, CONF_THRESH, CONF_THRESH, CONF_THRESH, CONF_THRESH, CONF_THRESH, CONF_THRESH}
#define LOC_FLOOR        0.250f   /* 0.08 -> 0.25: hoi vi tri vat KHONG co trong anh, 0.08 hay "bia" vi tri tu box diem thap
                                    (vd chuot khong co -> tra loi "ben phai" o cho dien thoai). tune_answer co hoi vat vang mat
                                    (30%): cau LOCATION khan200 0.786 (vang mat dung 0.761 -> 0.873), BTC val 0.960 -> 0.969 */

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
