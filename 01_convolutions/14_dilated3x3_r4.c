/* ============================================================================
 * [14] Dilated (atrous) Conv 3x3, rate = 4
 * ----------------------------------------------------------------------------
 * SHAPE: same 9 sample points, now spread even further apart. Covers a 9x9
 * area but touches only 9 cells -- they are 4 pixels apart in each direction.
 * From above: a very sparse scattered 3x3 pattern with large blank zones.
 *
 *   Input : [C][H][W]
 *   Weight: [C][3][3]                (9 weights, effective footprint 9x9)
 *   Output: [C][H][W]
 *
 * FLOPs ~ 2 * C * H * W * 9          (gaps cost nothing)
 *
 * DEMO: impulse + numbered kernel. The output values now land 4 cells apart.
 * Compare with [13] (rate=2): same kernel, wider holes, larger footprint.
 *
 * Build:  gcc 14_dilated3x3_r4.c -o 14_dilated3x3_r4 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K 3

static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s  [C=%d H=%d W=%d]\n", title, C, H, W);
    for (int c = 0; c < C; c++) {
        if (C > 1) {
            printf("  channel %d:\n", c);
        }
        for (int h = 0; h < H; h++) {
            printf("   ");
            for (int w = 0; w < W; w++) {
                printf("%6.1f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

void dilated3x3_r4(const float *in, const float *w, float *out,
                   int C, int H, int W, int rate)
{
    /* dilated conv: same 9 gather points as rate=2 but now rate=4 gives even
     * wider spacing -- effective receptive field is (K-1)*rate+1 = 9 wide */
    for (int c = 0; c < C; c++) {         /* c: output/input channel (depthwise) */
        for (int h = 0; h < H; h++) {     /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                        /* gap formula: center tap is (kh-1)*rate away from output row */
                        int ih = h + (kh - 1) * rate; /* input row: skips rate-1 rows between taps */
                        int iw = x + (kw - 1) * rate; /* input col: skips rate-1 cols between taps */
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            continue; /* outside image boundary = zero padding */
                        }
                        /* input flat index: [c][ih][iw] -> c*H*W + ih*W + iw */
                        sum += in[c*H*W + ih*W + iw] * w[c*K*K + kh*K + kw];
                    }
                }
                out[c*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 11, W = 11, rate = 4;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    in[5*W + 5] = 1.0f;                         /* impulse at center */
    for (int i = 0; i < K*K; i++) {
        w[i] = (float)(i + 1);                  /* numbered kernel 1..9 */
    }

    show("INPUT  (impulse at center)", in, C, H, W);
    show("KERNEL (numbered 1..9)", w, C, K, K);
    dilated3x3_r4(in, w, out, C, H, W, rate);
    show("OUTPUT (taps 4 cells apart -- wider holes than rate=2)", out, C, H, W);

    printf("[14] check: center out[5][5] = %.1f (expected 5.0)\n", out[5*W + 5]);

    free(in);
    free(w);
    free(out);
    return 0;
}
