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
 * MODEL THI DAU CHUNG KET: cam4B (init R3B, thay R3B 25ep + tro 45ep KD 2.0, cam_real1_giu_anh x6; INT8 outputs/cam4B).
 * Luu y: da train tren bo khan -> diem khan200/khan1075 khong dung de so. Anh chua thay (NMS 2 tang, nguong duoi,
 * LOC 0.10): xem README. Muon quay lai cam3R3B: git checkout b0ff117 -- firmware/main/models/ + nguong R3B
 * {0.35, 0.65, 0.45, 0.45, 0.70, 0.45, 0.45}. */
#define CONF_THRESH      0.45f
/* NMS 2 tang (nguoi dung chon: khong de 2 lop tren 1 vat): tang 1 cung lop IoU 0.30
 * + tang 2 khac lop YOLO_CROSS_NMS_IOU 0.60 trong logic.h (xoa box khac lop de khit >60% len cung 1 vat).
 * cam3R3B, nguong duoi: khan200 / khan1075 / BTC mau / real55 = 0.614 / 0.605 / 0.533 / 0.434
 * (gop lop 0.45: 0.614 / 0.603 / 0.532 / 0.434; theo lop 0.45 khong tang 2: 0.627 / 0.615 / 0.528 / 0.475). */
#define NMS_IOU          0.30f
#define NMS_AGNOSTIC     0        /* 0: per-class NMS 0.30 + tang 2 cross-class 0.60; 1: 1 tang gop lop */
/* nguong cua cam3R3B (nguoi dung chon dung cho ca cam3R3A_merge): bottle 0.35, cup 0.65, mouse 0.70.
 * cam3R3A_merge + nguong nay: BTC mau 0.512, real55 0.449, khan200 0.725
 * (nguong rieng cua ban do: {0.35,0.45,0.65,0.40,0.40,0.575,0.45} -> 0.531 / 0.422 / 0.731) */
#define CLASS_CONF_INIT  {0.60f, 0.50f, 0.70f, 0.45f, 0.70f, 0.65f, 0.65f}  /* nguoi dung chinh khi test cam4B: bottle 0.60, cup 0.50, phone 0.70, book 0.45, mouse 0.70, box 0.65, toy_car 0.65 */
#define LOC_FLOOR        0.100f   /* chung ket co DU do tren ban -> cau vi tri hau het hoi vat CO that.
                                    Neu de hoi nhieu vat vang mat -> 0.25 */

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
