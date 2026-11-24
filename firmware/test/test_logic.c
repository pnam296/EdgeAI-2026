/* Test hau xu ly tren may tinh (khong can board):
 *   gcc -I main test/test_logic.c main/logic.c -lm -o build_host/t && build_host/t
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "logic.h"

#define H 15
#define W 20
#define C (5 + NUM_CLASSES)

static void put(float *g, int y, int x, int cls, float obj)
{
    float *v = g + (y * W + x) * C;
    v[0] = obj; v[1] = v[2] = 0; v[3] = v[4] = 0;    /* offset 0.5, w=h=0.5 */
    v[5 + cls] = 10;
}

/* ---- YOLO: 1 muc stride 16 (15x20), reg_max 4, scale 0.1 zp 0 ---- */
#define RM 4
static int8_t ybox[H * W * 4 * RM], ycls[H * W * NUM_CLASSES];

static void yput(int y, int x, int cls, float logit, int dist)   /* box vuong canh 2*dist cell */
{
    int8_t *c = ycls + (y * W + x) * NUM_CLASSES;
    c[cls] = (int8_t)lroundf(logit / 0.1f);
    int8_t *b = ybox + (y * W + x) * 4 * RM;
    for (int s = 0; s < 4; s++) b[s * RM + dist] = 127;     /* phan bo DFL nhon tai dist */
}

static void test_yolo(void)
{
    memset(ybox, 0, sizeof ybox);
    memset(ycls, -100, sizeof ycls);                         /* logit -10 */
    yput(7, 2, 1, 3.0f, 2);       /* cup trai, score sig(3)=0.95 */
    yput(7, 3, 1, 2.0f, 2);       /* cup lech 1 cell -> IoU 0.6 -> NMS bo */
    yput(7, 17, 1, 3.0f, 1);      /* cup phai */
    yput(2, 10, 3, 1.0f, 2);      /* book giua, score 0.73 */
    yput(12, 10, 5, -2.0f, 2);    /* box diem 0.12 < conf -> bo */
    qtensor_t b = {ybox, 0.1f, 0, H, W, 4 * RM}, c = {ycls, 0.1f, 0, H, W, NUM_CLASSES};
    det_params_t p = {0.3f, 0.5f, 0};
    det_t d[16];
    int n = decode_yolo(&b, &c, 1, 320, 240, RM, &p, d, 16);
    for (int i = 0; i < n; i++)
        printf("  yolo %-6s %.3f [%.3f %.3f %.3f %.3f]\n", CLASS_NAMES[d[i].cls], d[i].score, d[i].x1, d[i].y1,
               d[i].x2, d[i].y2);
    assert(n == 3);
    assert(d[0].cls == 1 && fabsf(d[0].score - 0.9526f) < 1e-3f);
    assert(d[2].cls == 3);
    /* cup trai: tam (2.5, 7.5) cell * 16 px, dist ~2 cell (phan bo DFL gan one-hot) */
    float cx = (d[0].x1 + d[0].x2) / 2 * 320, cy = (d[0].y1 + d[0].y2) / 2 * 240;
    assert((fabsf(cx - 40) < 1 || fabsf(cx - 280) < 1) && fabsf(cy - 120) < 1);
    float wpx = (d[2].x2 - d[2].x1) * 320;
    assert(fabsf(wpx - 64) < 2);   /* 2 * dist(2) * stride 16 */

    p.conf = 0.1f;                 /* ha nguong -> them hop */
    assert(decode_yolo(&b, &c, 1, 320, 240, RM, &p, d, 16) == 4);
    p.conf = 0.8f;
    assert(decode_yolo(&b, &c, 1, 320, 240, RM, &p, d, 16) == 2);
    puts("decode_yolo OK");
}

int main(void)
{
    static float g[H * W * C];
    for (int i = 0; i < H * W; i++) g[i * C] = -10;   /* obj ~ 0 */
    put(g, 7, 2, 1, 5);          /* cup ben trai */
    put(g, 7, 3, 1, 4);          /* cup chong len cai tren -> NMS bo */
    put(g, 7, 17, 1, 5);         /* cup ben phai */
    put(g, 2, 10, 3, 5);         /* book giua */
    det_t d[64];
    int n = decode_grid(g, H, W, 0.3f, 0.5f, d, 64);
    assert(n == 3);
    assert(d[0].score >= d[1].score && d[1].score >= d[2].score);

    const char *vocab[] = {"bao", "co", "coc", "nhieu"};
    char q[] = "Co bao nhieu COC xyz?";   /* web da bo dau truoc khi gui */
    normalize_text(q);
    int32_t ids[24];
    tokenize(q, vocab, 4, ids, 24);
    assert(ids[0] == 3 && ids[1] == 2 && ids[2] == 5 && ids[3] == 4 && ids[4] == 1 && ids[5] == 0 && ids[23] == 0);

    char a[256];
    char q1[] = "co bao nhieu coc"; make_answer(0, q1, d, n, a, sizeof a); puts(a); assert(strstr(a, "2"));
    char q2[] = "co dien thoai khong"; make_answer(1, q2, d, n, a, sizeof a); puts(a); assert(strstr(a, "Không"));
    char q3[] = "sach nam o dau"; make_answer(3, q3, d, n, a, sizeof a); puts(a); assert(strstr(a, "giữa") && !strncmp(a, "Sách", 5));
    make_answer(2, q1, d, n, a, sizeof a); puts(a); assert(!strcmp(a, "Có 2 loại đối tượng: 2 cốc, 1 sách."));
    make_answer(4, q1, d, n, a, sizeof a); puts(a);
    char q4[] = "hop co khong phu"; make_answer(1, q4, d, 0, a, sizeof a); puts(a);
    char q5[] = "dem so sach va coc"; make_answer(0, q5, d, n, a, sizeof a); puts(a);
    assert(!strcmp(a, "Có 1 sách. Có 2 cốc."));
    char q6[] = "co binh nuoc nao khong"; make_answer(1, q6, d, n, a, sizeof a); puts(a); assert(strstr(a, "chai"));
    make_answer(2, q1, d, 0, a, sizeof a); puts(a);

    test_yolo();
    puts("OK");
    return 0;
}
