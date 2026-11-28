/* Logic thuan C (khong phu thuoc IDF) -> test duoc tren may host: test/test_logic.c, test/test_yolo_host.c */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CLASSES 7   /* thu tu = classes.txt cua dataset */
#define NUM_INTENTS 5   /* sorted(df.intent.unique()) */
#define MAX_SEQ     64  /* do dai cau toi da; do dai that lay tu input cua model TextCNN */

enum { INTENT_COUNT, INTENT_EXIST, INTENT_LIST, INTENT_LOCATION, INTENT_OUT_OF_SCOPE };

extern const char *const CLASS_NAMES[NUM_CLASSES];
extern const char *const INTENT_NAMES[NUM_INTENTS];

typedef struct { int cls; float score, x1, y1, x2, y2; } det_t;   /* toa do chuan hoa 0..1 */

typedef struct {
    float conf;       /* nguong score */
    float iou;        /* nguong IoU cua NMS */
    int agnostic;     /* 1 = NMS khong phan biet lop (bot box trung lop khac tren cung 1 vat) */
} det_params_t;

/* ---- Head grid kieu baseline (1 output NHWC [H][W][5+NUM_CLASSES]) ---- */
int decode_grid(const float *g, int H, int W, float conf, float iou, det_t *out, int max);

/* ---- Head YOLO anchor-free (Ultralytics v8/11, DFL) ----
 * Moi muc (stride) co 2 tensor int8 NHWC: box [H][W][4*reg_max] (logit DFL, canh l,t,r,b)
 * va cls [H][W][NUM_CLASSES] (logit). stride = img_h / H. Score = sigmoid(logit lop lon nhat). */
typedef struct {
    const int8_t *data;
    float scale;
    int zp;
    int H, W, C;
} qtensor_t;

int decode_yolo(const qtensor_t *box, const qtensor_t *cls, int nlevels, int img_w, int img_h, int reg_max,
                const det_params_t *p, det_t *out, int max);

/* He so tam voi cua muc cuoi (P4): khoang cach DFL nhan them he so nay (tools/yolo_reach.py).
 * PHAI KHOP luc train: model train voi yolo_reach.apply((1, 1, 1.5)) -> 1.5f (box toi da ~144px).
 * Model cu (train khong co yolo_reach) -> 1.0f. */
#ifndef YOLO_DIST_SCALE_LAST
#define YOLO_DIST_SCALE_LAST 1.5f
#endif

/* Tang 2 NMS khac lop (chi chay khi p->agnostic == 0 va < 1.0f):
 * xoa box khac lop de khit len cung 1 vat (IoU > 0.60) ma van giu 2 vat khac lop dat sat nhau. */
#ifndef YOLO_CROSS_NMS_IOU
#define YOLO_CROSS_NMS_IOU 0.60f
#endif

/* lowercase, ky tu khong phai [a-z0-9] -> ' ' (giong normalize() cua pipeline: re.sub(r"[^a-z0-9 ]", " ")) */
void normalize_text(char *s);

/* vocab[k] co id k+2; 0 = pad, 1 = unknown. ids co seq_len phan tu (<= MAX_SEQ), thieu thi dem 0 */
void tokenize(const char *norm, const char *const *vocab, int nv, int32_t *ids, int seq_len);

/* Nguong cho cau tra loi (chinh bang tools/tune_answer.py theo do chinh xac cau tra loi, khong theo mAP):
 *  cls_conf[c]: box lop c duoc tinh cho COUNT/EXIST/LIST/LOCATION khi score >= cls_conf[c]
 *  loc_floor  : LOCATION hoi lop c ma khong co box nao >= cls_conf[c] -> lay box lop c diem cao nhat
 *               >= loc_floor (cau "X o dau" ngam bao X co trong anh). Dat >= 1 de tat.
 * Detection phai decode voi p->conf <= answer_floor(ap) de giu du box cho ca hai nguong. */
typedef struct {
    float cls_conf[NUM_CLASSES];
    float loc_floor;
} answer_params_t;

float answer_floor(const answer_params_t *ap);

/* Giu box co score >= cls_conf cua lop no (giu thu tu). out co the trung d. ap NULL = giu het. */
int filter_by_class_conf(const det_t *d, int n, const answer_params_t *ap, det_t *out);

/* Cau tra loi tieng Viet theo intent + ket qua detection.
 * d: box sau NMS o nguong san (sort score giam dan); ap: nguong lop + du phong LOCATION (NULL = dung d nguyen).
 * Tra ve intent thuc te (COUNT/EXIST/LOCATION khong bat duoc ten vat -> INTENT_OUT_OF_SCOPE). */
/* cac lop duoc nhac toi trong cau da normalize (theo thu tu xuat hien), tra ve so lop - cho cong cu may tinh */
int logic_find_classes(const char *norm, int *cls);

int make_answer(int intent, const char *norm_q, const det_t *d, int n, const answer_params_t *ap, char *out,
                size_t sz);

#ifdef __cplusplus
}
#endif
