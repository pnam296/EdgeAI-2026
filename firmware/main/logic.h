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

/* lowercase, ky tu khong phai [a-z0-9] -> ' ' (giong normalize() cua pipeline: re.sub(r"[^a-z0-9 ]", " ")) */
void normalize_text(char *s);

/* vocab[k] co id k+2; 0 = pad, 1 = unknown. ids co seq_len phan tu (<= MAX_SEQ), thieu thi dem 0 */
void tokenize(const char *norm, const char *const *vocab, int nv, int32_t *ids, int seq_len);

/* Cau tra loi tieng Viet theo intent + ket qua detection (d da sort theo score giam dan) */
void make_answer(int intent, const char *norm_q, const det_t *d, int n, char *out, size_t sz);

#ifdef __cplusplus
}
#endif
