#pragma once
#include "logic.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IMG_W 320
#define IMG_H 240
#define IMG_BYTES (IMG_W * IMG_H * 3)

typedef struct {
    int ms_fill, ms_invoke, ms_decode;   /* chia nho thoi gian detect (de toi uu) */
} det_timing_t;

int infer_init(void);                                                   /* 0 = ok */
const char *infer_det_kind(void);                                       /* "yolo" | "grid" | "none" */
/* rgb 320x240 RGB888 -> so box (sort theo score), <0 loi. p = NULL dung nguong mac dinh trong model_config.h */
int infer_detect(const uint8_t *rgb, const det_params_t *p, det_t *out, int max, det_timing_t *t);
int infer_intent(const char *norm_q);                                   /* index INTENT_NAMES, <0 loi */
void infer_default_params(det_params_t *p);

#ifdef __cplusplus
}
#endif
