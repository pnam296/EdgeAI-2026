/* Chay decode_yolo (code C THAT cua firmware) tren output int8 do tools/verify_det.py dump ra.
 *   gcc -O2 -I main test/yolo_host.c main/logic.c -lm -o build_host/yolo_host
 *   build_host/yolo_host dump.bin conf iou agnostic > dets.txt
 * dump.bin: int32 n_img, nlevels, reg_max, img_w, img_h; roi moi anh, moi muc: [box, cls] voi moi tensor
 *           int32 H, W, C; float scale; int32 zp; int8 data[H*W*C]
 * stdout: moi dong "img cls score x1 y1 x2 y2" (toa do chuan hoa) */
#include <stdio.h>
#include <stdlib.h>
#include "logic.h"

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int read_t(FILE *f, qtensor_t *t)
{
    int32_t h[3], zp; float sc;
    if (!rd(f, h, sizeof h) || !rd(f, &sc, 4) || !rd(f, &zp, 4)) return 0;
    size_t n = (size_t)h[0] * h[1] * h[2];
    int8_t *d = malloc(n);
    if (!d || !rd(f, d, n)) return 0;
    *t = (qtensor_t){d, sc, zp, h[0], h[1], h[2]};
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "dung: yolo_host dump.bin [conf iou agnostic]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    det_params_t p = {argc > 2 ? (float)atof(argv[2]) : 0.25f, argc > 3 ? (float)atof(argv[3]) : 0.7f,
                      argc > 4 ? atoi(argv[4]) : 0};
    int32_t hd[5];
    if (!rd(f, hd, sizeof hd)) return 1;
    int nimg = hd[0], nl = hd[1], rm = hd[2], iw = hd[3], ih = hd[4];
    static det_t d[300];
    for (int i = 0; i < nimg; i++) {
        qtensor_t b[8], c[8];
        for (int l = 0; l < nl; l++)
            if (!read_t(f, &b[l]) || !read_t(f, &c[l])) { fprintf(stderr, "dump hong o anh %d\n", i); return 1; }
        int n = decode_yolo(b, c, nl, iw, ih, rm, &p, d, 300);
        for (int k = 0; k < n; k++)
            printf("%d %d %.5f %.5f %.5f %.5f %.5f\n", i, d[k].cls, d[k].score, d[k].x1, d[k].y1, d[k].x2, d[k].y2);
        for (int l = 0; l < nl; l++) { free((void *)b[l].data); free((void *)c[l].data); }
    }
    fclose(f);
    return 0;
}
