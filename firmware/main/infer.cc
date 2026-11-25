#include "infer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "model_config.h"

static const char *TAG = "infer";

static tflite::MicroInterpreter *s_det, *s_txt;
static int8_t s_lut[256];             /* pixel 0..255 -> int8 da quantize */
static int s_seq_len;                 /* do dai cau cua TextCNN (doc tu input model) */

/* Kieu head detection, tu nhan theo output cua model */
enum { DET_NONE, DET_GRID, DET_YOLO };
static int s_kind = DET_NONE;
#define MAX_LEVELS 4
static int s_nl;                                   /* so muc YOLO */
static int s_box_idx[MAX_LEVELS], s_cls_idx[MAX_LEVELS];   /* chi so output, sap theo H giam dan (P2, P3, P4) */

static tflite::MicroInterpreter *load(const unsigned char *data, int arena_kb, const char *name)
{
    static tflite::MicroMutableOpResolver<RESOLVER_SIZE> ops;
    static bool ops_ready;
    if (!ops_ready) { REGISTER_OPS(ops); ops_ready = true; }
    const tflite::Model *model = tflite::GetModel(data);
    if (model->version() != TFLITE_SCHEMA_VERSION) { ESP_LOGE(TAG, "%s: sai schema version", name); return nullptr; }
    uint8_t *arena = (uint8_t *)heap_caps_aligned_alloc(16, arena_kb * 1024, MALLOC_CAP_SPIRAM);
    if (!arena) { ESP_LOGE(TAG, "%s: khong cap phat duoc %d KB PSRAM", name, arena_kb); return nullptr; }
    auto *it = new tflite::MicroInterpreter(model, ops, arena, arena_kb * 1024);
    if (it->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "%s: AllocateTensors loi (op thieu -> chay lai tools/*.py de sinh ops_gen.h; "
                      "arena thieu -> tang *_ARENA_KB)", name);
        return nullptr;
    }
    ESP_LOGI(TAG, "%s OK | arena dung %u / %d KB", name, (unsigned)it->arena_used_bytes() / 1024, arena_kb);
    return it;
}

/* Nhan dien head: 1 output C = 5+NC -> grid baseline; cap (4*REG_MAX, NC) cung HxW -> YOLO */
static int detect_kind(void)
{
    int n = s_det->outputs_size();
    if (n == 1) {
        TfLiteTensor *o = s_det->output(0);
        if (o->dims->size == 4 && o->dims->data[3] == 5 + NUM_CLASSES) return DET_GRID;
    }
    if (n % 2 || n / 2 > MAX_LEVELS) return DET_NONE;
    int hs[MAX_LEVELS], nl = 0;
    for (int i = 0; i < n; i++) {
        TfLiteTensor *o = s_det->output(i);
        if (o->dims->size != 4 || o->type != kTfLiteInt8) return DET_NONE;
        int h = o->dims->data[1], c = o->dims->data[3];
        if (c != 4 * YOLO_REG_MAX && c != NUM_CLASSES) {
            ESP_LOGE(TAG, "output %d co %d kenh: can %d (box, reg_max %d) hoac %d (lop)", i, c, 4 * YOLO_REG_MAX,
                     YOLO_REG_MAX, NUM_CLASSES);
            return DET_NONE;
        }
        int l = 0;
        while (l < nl && hs[l] != h) l++;
        if (l == nl) { if (nl == MAX_LEVELS) return DET_NONE; hs[nl] = h; s_box_idx[nl] = s_cls_idx[nl] = -1; nl++; }
        (c == NUM_CLASSES ? s_cls_idx : s_box_idx)[l] = i;
    }
    for (int i = 0; i < nl; i++)      /* sap theo H giam dan */
        for (int j = i + 1; j < nl; j++)
            if (hs[j] > hs[i]) {
                int t = hs[i]; hs[i] = hs[j]; hs[j] = t;
                t = s_box_idx[i]; s_box_idx[i] = s_box_idx[j]; s_box_idx[j] = t;
                t = s_cls_idx[i]; s_cls_idx[i] = s_cls_idx[j]; s_cls_idx[j] = t;
            }
    for (int l = 0; l < nl; l++) {
        if (s_box_idx[l] < 0 || s_cls_idx[l] < 0) return DET_NONE;
        ESP_LOGI(TAG, "  muc %d: %dx%d stride %d (box=out%d, cls=out%d)", l, hs[l],
                 s_det->output(s_cls_idx[l])->dims->data[2], IMG_H / hs[l], s_box_idx[l], s_cls_idx[l]);
    }
    s_nl = nl;
    return DET_YOLO;
}

int infer_init(void)
{
    s_det = load(DET_MODEL_DATA, DET_ARENA_KB, "detection");
    s_txt = load(TXT_MODEL_DATA, TXT_ARENA_KB, "textcnn");
    if (!s_det || !s_txt) { s_det = s_txt = nullptr; return -1; }

    TfLiteTensor *in = s_det->input(0);
    if (in->type != kTfLiteInt8 || in->bytes != IMG_BYTES) {
        ESP_LOGE(TAG, "detection: can input int8 1x240x320x3");
        s_det = s_txt = nullptr;
        return -1;
    }
    s_kind = detect_kind();
    if (s_kind == DET_NONE) {
        ESP_LOGE(TAG, "detection: khong nhan ra head (grid 1 output ...x%d, hoac YOLO cap box %d / cls %d kenh)",
                 5 + NUM_CLASSES, 4 * YOLO_REG_MAX, NUM_CLASSES);
        s_det = s_txt = nullptr;
        return -1;
    }
    ESP_LOGI(TAG, "detection head: %s", infer_det_kind());
    for (int v = 0; v < 256; v++)
        s_lut[v] = (int8_t)fminf(127, fmaxf(-128, roundf(v / 255.f / in->params.scale + in->params.zero_point)));

    TfLiteTensor *ti = s_txt->input(0), *to = s_txt->output(0);
    s_seq_len = ti->dims->data[ti->dims->size - 1];
    if (to->dims->data[to->dims->size - 1] != NUM_INTENTS || s_seq_len > MAX_SEQ) {
        ESP_LOGE(TAG, "textcnn: output can %d intent, input <= %d token", NUM_INTENTS, MAX_SEQ);
        s_det = s_txt = nullptr;
        return -1;
    }
    ESP_LOGI(TAG, "vocab %d tu, cau %d token", VOCAB_LEN, s_seq_len);
    return 0;
}

const char *infer_det_kind(void)
{
    return s_kind == DET_YOLO ? "yolo" : s_kind == DET_GRID ? "grid" : "none";
}

void answer_default_params(answer_params_t *ap)
{
    static const float cls_conf[NUM_CLASSES] = CLASS_CONF_INIT;
    for (int c = 0; c < NUM_CLASSES; c++) ap->cls_conf[c] = cls_conf[c];
    ap->loc_floor = LOC_FLOOR;
}

void infer_default_params(det_params_t *p)
{
    answer_params_t ap;
    answer_default_params(&ap);
    p->conf = answer_floor(&ap);
    p->iou = NMS_IOU;
    p->agnostic = NMS_AGNOSTIC;
}

static float deq(const TfLiteTensor *t, int i)
{
    return t->type == kTfLiteInt8 ? (t->data.int8[i] - t->params.zero_point) * t->params.scale : t->data.f[i];
}

static qtensor_t qt(const TfLiteTensor *t)
{
    return qtensor_t{t->data.int8, t->params.scale, (int)t->params.zero_point,
                     t->dims->data[1], t->dims->data[2], t->dims->data[3]};
}

int infer_detect(const uint8_t *rgb, const det_params_t *p, det_t *out, int max, det_timing_t *tm)
{
    if (!s_det) return -1;
    det_params_t def;
    if (!p) { infer_default_params(&def); p = &def; }
    int64_t t0 = esp_timer_get_time();
    TfLiteTensor *in = s_det->input(0);
    int8_t *dst = in->data.int8;
    for (int i = 0; i < IMG_BYTES; i++) dst[i] = s_lut[rgb[i]];
    int64_t t1 = esp_timer_get_time();
    if (s_det->Invoke() != kTfLiteOk) return -1;
    int64_t t2 = esp_timer_get_time();
    int k;
    if (s_kind == DET_YOLO) {
        qtensor_t box[MAX_LEVELS], cls[MAX_LEVELS];
        for (int l = 0; l < s_nl; l++) {
            box[l] = qt(s_det->output(s_box_idx[l]));
            cls[l] = qt(s_det->output(s_cls_idx[l]));
        }
        k = decode_yolo(box, cls, s_nl, IMG_W, IMG_H, YOLO_REG_MAX, p, out, max);
    } else {
        TfLiteTensor *o = s_det->output(0);
        int H = o->dims->data[1], W = o->dims->data[2], n = H * W * (5 + NUM_CLASSES);
        float *g = (float *)heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM);
        if (!g) return -1;
        for (int i = 0; i < n; i++) g[i] = deq(o, i);
        k = decode_grid(g, H, W, p->conf, p->iou, out, max);
        free(g);
    }
    int64_t t3 = esp_timer_get_time();
    if (tm) *tm = det_timing_t{(int)((t1 - t0) / 1000), (int)((t2 - t1) / 1000), (int)((t3 - t2) / 1000)};
    return k;
}

int infer_intent(const char *norm_q)
{
    if (!s_txt) return -1;
    int32_t ids[MAX_SEQ];
    tokenize(norm_q, VOCAB_DATA, VOCAB_LEN, ids, s_seq_len);
    TfLiteTensor *in = s_txt->input(0);
    for (int i = 0; i < s_seq_len; i++) {
        if (in->type == kTfLiteInt64) in->data.i64[i] = ids[i];
        else if (in->type == kTfLiteInt32) in->data.i32[i] = ids[i];
        else if (in->type == kTfLiteFloat32) in->data.f[i] = ids[i];
        else { ESP_LOGE(TAG, "textcnn: kieu input %d chua ho tro", in->type); return -1; }
    }
    if (s_txt->Invoke() != kTfLiteOk) return -1;
    TfLiteTensor *o = s_txt->output(0);
    int best = 0;
    for (int k = 1; k < NUM_INTENTS; k++) if (deq(o, k) > deq(o, best)) best = k;
    return best;
}
