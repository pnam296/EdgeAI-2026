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
 *  LOC_FLOOR      : LOCATION hoi lop X ma khong co box X du nguong -> lay box X diem cao nhat >= LOC_FLOOR */
 * Gia tri duoi (model cam1 yolo_cam_best.pt, P4 x1.5, INT8; tune_answer.py tren BTC val + Roboflow mo phong camera):
 * tong 0.869 (C+E 0.943, LIST 0.635, LOC 0.978); BTC 0.892, RF 0.846. Model cu cung bo val: tong 0.617 (RF 0.319).
 * Script chon toy_car 0.275 nhung loi chi +0.003 (khop qua val, de bao nham board/module tren anh that) -> giu 0.475;
 * chinh lai tung lop khi co tap anh camera that da gan nhan. */
#define CONF_THRESH      0.475f
#define NMS_IOU          0.40f
#define NMS_AGNOSTIC     1        /* 1: bo box lop khac trung len cung 1 vat (giam false positive) */
#define CLASS_CONF_INIT  {0.475f, 0.475f, 0.475f, 0.475f, 0.625f, 0.625f, 0.475f}
#define LOC_FLOOR        0.080f

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
