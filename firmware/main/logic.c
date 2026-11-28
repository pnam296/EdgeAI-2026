#include "logic.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const CLASS_NAMES[NUM_CLASSES] = {"bottle", "cup", "phone", "book", "mouse", "box", "toy_car"};
const char *const INTENT_NAMES[NUM_INTENTS] = {"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"};

static const char *const VN[NUM_CLASSES] = {"chai", "cốc", "điện thoại", "sách", "chuột", "hộp", "xe đồ chơi"};
static const char *const VN_CAP[NUM_CLASSES] = {"Chai", "Cốc", "Điện thoại", "Sách", "Chuột", "Hộp", "Xe đồ chơi"};
/* Tu khoa (khong dau) de biet cau hoi nhac toi lop nao. Lay theo IN_SCOPE_NAMES cua pipeline intent.
 * Cum nhieu tu dat truoc tu don cua cung lop. */
#define MAX_KW 16
static const char *const KW[NUM_CLASSES][MAX_KW] = {
    {"binh nuoc", "binh giu nhiet", "chai", "binh", "lavie", "aquafina"},
    {"coc", "ly", "tach", "mug"},
    {"dien thoai", "di dong", "smartphone", "thoai", "dtdd", "dt", "iphone", "samsung", "oppo", "xiaomi", "nokia"},
    {"sach", "quyen", "cuon", "vo", "so tay", "tap", "truyen", "giao trinh"},
    {"chuot"},
    {"thung giay", "thung carton", "khan giay", "hop", "thung", "carton"},
    {"o to do choi", "oto do choi", "xe do choi", "hot wheels", "oto", "o to", "xe"},
};

/* tu thuong gap gan giong tu khoa (minh~binh, chao~chai...) -> khong bao gio khop gan dung */
static int is_stop(const char *w, size_t lw)
{
    static const char *const S[] = {"minh", "chao", "chan", "nhieu", "thoi", "hinh", "trong", "khong", "nhung", "thay"};
    for (size_t i = 0; i < sizeof S / sizeof *S; i++)
        if (strlen(S[i]) == lw && !strncmp(S[i], w, lw)) return 1;
    return 0;
}

/* khop gan dung (gõ sai 1 ky tu: thua/thieu/sai/dao 2 ky tu ke nhau) - CHI dung khi khong khop chinh xac lop nao.
 * Chi voi tu khoa 1 tu, dai >= 4 (tranh "dt"~"de", "ly"~"la"). */
static int near1(const char *a, size_t la, const char *b, size_t lb)
{
    if (la > lb + 1 || lb > la + 1) return 0;
    size_t i = 0;
    while (i < la && i < lb && a[i] == b[i]) i++;
    if (i == la && i == lb) return 1;
    if (la == lb) {
        if (!strncmp(a + i + 1, b + i + 1, la - i - 1)) return 1;                       /* sai 1 ky tu */
        return i + 1 < la && a[i] == b[i + 1] && a[i + 1] == b[i] &&
               !strncmp(a + i + 2, b + i + 2, la - i - 2);                                /* dao 2 ky tu */
    }
    return la > lb ? !strncmp(a + i + 1, b + i, lb - i) : !strncmp(a + i, b + i + 1, la - i);   /* thua/thieu */
}

static float sig(float x) { return 1.f / (1.f + expf(-x)); }
static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

static int by_score(const void *a, const void *b)
{
    float d = ((const det_t *)b)->score - ((const det_t *)a)->score;
    return (d > 0) - (d < 0);
}

static float box_iou(const det_t *a, const det_t *b)
{
    float w = fminf(a->x2, b->x2) - fmaxf(a->x1, b->x1);
    float h = fminf(a->y2, b->y2) - fmaxf(a->y1, b->y1);
    if (w <= 0 || h <= 0) return 0;
    float inter = w * h;
    float u = (a->x2 - a->x1) * (a->y2 - a->y1) + (b->x2 - b->x1) * (b->y2 - b->y1) - inter;
    return u > 0 ? inter / u : 0;
}

/* sort theo score roi NMS tai cho, tra ve so box giu lai */
static int nms(det_t *d, int n, float iou, int agnostic)
{
    qsort(d, n, sizeof(det_t), by_score);
    int keep = 0;
    for (int i = 0; i < n; i++) {
        int ok = 1;
        for (int j = 0; j < keep && ok; j++)
            if ((agnostic || d[j].cls == d[i].cls) && box_iou(&d[j], &d[i]) > iou) ok = 0;
        if (ok) d[keep++] = d[i];
    }
    return keep;
}

/* = decode_boxes + softmax + batched_nms trong baseline.ipynb */
int decode_grid(const float *g, int H, int W, float conf, float iou, det_t *out, int max)
{
    const int C = 5 + NUM_CLASSES;
    int n = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float *v = g + (y * W + x) * C;
            int best = 0;
            for (int k = 1; k < NUM_CLASSES; k++) if (v[5 + k] > v[5 + best]) best = k;
            float sum = 0;
            for (int k = 0; k < NUM_CLASSES; k++) sum += expf(v[5 + k] - v[5 + best]);
            float score = sig(v[0]) / sum;
            if (score < conf || n >= max) continue;
            float cx = (sig(v[1]) + x) / W, cy = (sig(v[2]) + y) / H, w = sig(v[3]), h = sig(v[4]);
            out[n++] = (det_t){best, score, clamp01(cx - w / 2), clamp01(cy - h / 2),
                               clamp01(cx + w / 2), clamp01(cy + h / 2)};
        }
    return nms(out, n, iou, 0);
}

/* ------------------------------------------------------------------ YOLO (DFL) */
#define MAX_CAND 512
static det_t s_cand[MAX_CAND];

/* ky vong cua phan bo DFL: sum_j j * softmax(v)_j */
static float dfl(const int8_t *v, int reg_max, float scale)
{
    int mx = v[0];
    for (int j = 1; j < reg_max; j++) if (v[j] > mx) mx = v[j];
    float s = 0, e = 0;
    for (int j = 0; j < reg_max; j++) {
        float w = expf((v[j] - mx) * scale);
        s += w;
        e += w * j;
    }
    return e / s;
}

int decode_yolo(const qtensor_t *box, const qtensor_t *cls, int nlevels, int img_w, int img_h, int reg_max,
                const det_params_t *p, det_t *out, int max)
{
    int n = 0, worst = -1;
    float conf = p->conf < 1e-4f ? 1e-4f : p->conf > 0.9999f ? 0.9999f : p->conf;
    float logit_t = logf(conf / (1 - conf));
    for (int l = 0; l < nlevels; l++) {
        const qtensor_t *bt = &box[l], *ct = &cls[l];
        const float stride = (float)img_h / ct->H;
        const float ks = (l == nlevels - 1) ? YOLO_DIST_SCALE_LAST : 1.0f; /* tam voi muc cuoi */
        /* so sanh tren int8: v >= q_t  <=>  (v - zp) * scale >= logit_t (scale > 0) */
        float qf = ceilf(logit_t / ct->scale + ct->zp);
        int q_t = qf < -128 ? -128 : qf > 128 ? 128 : (int)qf;
        for (int y = 0; y < ct->H; y++)
            for (int x = 0; x < ct->W; x++) {
                const int8_t *c = ct->data + (y * ct->W + x) * ct->C;
                int best = 0;
                for (int k = 1; k < NUM_CLASSES; k++) if (c[k] > c[best]) best = k;
                if (c[best] < q_t) continue;
                float score = sig((c[best] - ct->zp) * ct->scale);
                const int8_t *b = bt->data + (y * bt->W + x) * bt->C;
                float d[4];
                for (int s = 0; s < 4; s++) d[s] = dfl(b + s * reg_max, reg_max, bt->scale) * ks;
                /* bo zero-point: softmax bat bien khi cong hang so, nen chi can scale */
                float ax = x + 0.5f, ay = y + 0.5f;
                det_t t = {best, score,
                           clamp01((ax - d[0]) * stride / img_w), clamp01((ay - d[1]) * stride / img_h),
                           clamp01((ax + d[2]) * stride / img_w), clamp01((ay + d[3]) * stride / img_h)};
                if (n < MAX_CAND) {
                    s_cand[n++] = t;
                    worst = -1;
                } else {   /* day: thay box diem thap nhat */
                    if (worst < 0) {
                        worst = 0;
                        for (int i = 1; i < n; i++) if (s_cand[i].score < s_cand[worst].score) worst = i;
                    }
                    if (score > s_cand[worst].score) { s_cand[worst] = t; worst = -1; }
                }
            }
    }
    int k = nms(s_cand, n, p->iou, p->agnostic);
    if (!p->agnostic && YOLO_CROSS_NMS_IOU < 1.0f)
        k = nms(s_cand, k, YOLO_CROSS_NMS_IOU, 1);
    if (k > max) k = max;
    memcpy(out, s_cand, k * sizeof(det_t));
    return k;
}

/* ------------------------------------------------------------------ text */
void normalize_text(char *s)
{
    for (; *s; s++) {
        char c = *s;
        if (c >= 'A' && c <= 'Z') *s = c - 'A' + 'a';
        else if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) *s = ' ';
    }
}

void tokenize(const char *norm, const char *const *vocab, int nv, int32_t *ids, int seq_len)
{
    int n = 0;
    if (seq_len > MAX_SEQ) seq_len = MAX_SEQ;
    memset(ids, 0, seq_len * sizeof(int32_t));
    while (*norm && n < seq_len) {
        while (*norm == ' ') norm++;
        size_t len = strcspn(norm, " ");
        if (!len) break;
        int id = 1, lo = 0, hi = nv - 1;   /* binary search tren vocab da sort -> O(log nv) cho vocab lon */
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            const char *w = vocab[mid];
            int cmp = strncmp(w, norm, len);
            if (cmp == 0) cmp = (w[len] != '\0');
            if (cmp == 0) { id = mid + 2; break; }
            if (cmp < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        ids[n++] = id;
        norm += len;
    }
}

/* vi tri xuat hien dau tien cua cum tu w (dung ranh gioi tu), -1 neu khong co */
static int find_word(const char *norm, const char *w)
{
    size_t len = strlen(w);
    for (const char *p = norm; (p = strstr(p, w)); p++)
        if ((p == norm || p[-1] == ' ') && (p[len] == ' ' || p[len] == 0)) return (int)(p - norm);
    return -1;
}

/* cac lop duoc nhac toi trong cau, theo thu tu xuat hien. Tra ve so lop */
static int find_classes(const char *norm, int *cls)
{
    int pos[NUM_CLASSES], n = 0;
    for (int c = 0; c < NUM_CLASSES; c++) {
        int best = -1;
        for (int k = 0; k < MAX_KW && KW[c][k]; k++) {
            int p = find_word(norm, KW[c][k]);
            if (p >= 0 && (best < 0 || p < best)) best = p;
        }
        if (best >= 0) { cls[n] = c; pos[n] = best; n++; }
    }
    if (!n) {   /* khong khop chinh xac: thu khop gan dung tung tu trong cau (gõ sai 1 ky tu) */
        for (int c = 0; c < NUM_CLASSES; c++) {
            int best = -1;
            for (int k = 0; k < MAX_KW && KW[c][k] && best < 0; k++) {
                const char *kw = KW[c][k];
                size_t lk = strlen(kw);
                if (lk < 4 || strchr(kw, ' ')) continue;
                for (const char *p = norm; *p && best < 0;) {
                    while (*p == ' ') p++;
                    size_t lw = strcspn(p, " ");
                    if (lw >= 4 && !is_stop(p, lw) && near1(p, lw, kw, lk)) best = (int)(p - norm);
                    p += lw;
                }
            }
            if (best >= 0) { cls[n] = c; pos[n] = best; n++; }
        }
    }
    for (int i = 1; i < n; i++)   /* insertion sort theo vi tri */
        for (int j = i; j > 0 && pos[j] < pos[j - 1]; j--) {
            int t = pos[j]; pos[j] = pos[j - 1]; pos[j - 1] = t;
            t = cls[j]; cls[j] = cls[j - 1]; cls[j - 1] = t;
        }
    return n;
}

static const char *where(const det_t *b)
{
    float cx = (b->x1 + b->x2) / 2, cy = (b->y1 + b->y2) / 2;
    static const char *const T[3][3] = {
        {"phía trên bên trái", "phía trên ở giữa", "phía trên bên phải"},
        {"bên trái", "ở giữa", "bên phải"},
        {"phía dưới bên trái", "phía dưới ở giữa", "phía dưới bên phải"},
    };
    int col = cx < 1.f / 3 ? 0 : cx < 2.f / 3 ? 1 : 2;
    int row = cy < 1.f / 3 ? 0 : cy < 2.f / 3 ? 1 : 2;
    return T[row][col];
}

#define APPEND(...) do { if (p < sz) p += snprintf(out + p, sz - p, __VA_ARGS__); } while (0)

float answer_floor(const answer_params_t *ap)
{
    float f = ap->loc_floor;
    for (int c = 0; c < NUM_CLASSES; c++)
        if (ap->cls_conf[c] < f) f = ap->cls_conf[c];
    return f;
}

int filter_by_class_conf(const det_t *d, int n, const answer_params_t *ap, det_t *out)
{
    int k = 0;
    for (int i = 0; i < n; i++)
        if (!ap || d[i].score >= ap->cls_conf[d[i].cls]) out[k++] = d[i];
    return k;
}

#define MAX_ANS_DETS 128

void make_answer(int intent, const char *norm_q, const det_t *d_all, int n_all, const answer_params_t *ap,
                 char *out, size_t sz)
{
    static det_t f[MAX_ANS_DETS];   /* static: khong ton stack task httpd */
    if (n_all > MAX_ANS_DETS) n_all = MAX_ANS_DETS;
    const int n = filter_by_class_conf(d_all, n_all, ap, f);
    const det_t *d = f;
    size_t p = 0;
    out[0] = 0;
    const char *name = INTENT_NAMES[intent];
    if (!strcmp(name, "OUT_OF_SCOPE")) {
        APPEND("Câu hỏi nằm ngoài phạm vi.");
        return;
    }
    if (!strcmp(name, "LIST")) {
        int cnt[NUM_CLASSES] = {0}, k = 0;
        for (int i = 0; i < n; i++) k += !cnt[d[i].cls]++;
        if (!k) { APPEND("Không có đối tượng nào trong ảnh."); return; }
        APPEND("Có %d loại đối tượng", k);
        const char *sep = ":";
        for (int c = 0; c < NUM_CLASSES; c++)
            if (cnt[c]) { APPEND("%s %d %s", sep, cnt[c], VN[c]); sep = ","; }
        APPEND(".");
        return;
    }
    int cls[NUM_CLASSES];
    int m = find_classes(norm_q, cls);
    if (!m) {
        APPEND("Không xác định được đối tượng trong câu hỏi.");
        return;
    }
    for (int i = 0; i < m; i++) {
        int c = cls[i], cnt = 0, best = -1;
        for (int j = 0; j < n; j++)
            if (d[j].cls == c) { cnt++; if (best < 0) best = j; }   /* d da sort theo score */
        if (i) APPEND(" ");
        if (!strcmp(name, "COUNT")) {
            APPEND("Có %d %s.", cnt, VN[c]);
        } else if (!strcmp(name, "EXIST")) {
            APPEND(cnt ? "Có %s trong ảnh." : "Không có %s trong ảnh.", VN[c]);
        } else if (best < 0) {   /* LOCATION: khong co box du nguong lop -> du phong o nguong loc_floor */
            int fb = -1;
            for (int j = 0; ap && j < n_all && fb < 0; j++)   /* d_all da sort theo score */
                if (d_all[j].cls == c && d_all[j].score >= ap->loc_floor) fb = j;
            if (fb >= 0) APPEND("%s nằm %s.", VN_CAP[c], where(&d_all[fb]));
            else APPEND("Không thấy %s trong ảnh.", VN[c]);
        } else {
            APPEND("%s nằm %s", VN_CAP[c], where(&d[best]));
            for (int j = best + 1, extra = 0; j < n && extra < 2; j++)   /* them toi da 2 vat cung lop */
                if (d[j].cls == c) { APPEND(", %s", where(&d[j])); extra++; }
            APPEND(".");
        }
    }
}

/* cho cong cu tren may tinh (tools/eval_qa.py) goi dung ham tim ten vat cua firmware */
int logic_find_classes(const char *norm, int *cls) { return find_classes(norm, cls); }
