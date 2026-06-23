/* ============================================================================
 * [13] Dilated (atrous) Conv 3x3, rate = 2
 * ----------------------------------------------------------------------------
 * SHAPE: still 9 sample points in a 3x3 arrangement, but with GAPS -- every
 * other pixel is skipped. The 9 taps cover a 5x5 footprint while touching only
 * 9 of the 25 cells. The trick: tap offset = (k - 1) * rate instead of (k - 1).
 * Enlarges the receptive field WITHOUT extra parameters or downsampling.
 *
 *   Input : [C][H][W]
 *   Weight: [C][3][3]                (still only 9 weights, just spaced out)
 *   Output: [C][H][W]                (effective footprint 5x5 at rate 2)
 *
 * FLOPs ~ 2 * C * H * W * 9          (same as a normal 3x3 -- gaps are free)
 *
 * DEMO: impulse + numbered kernel 1..9. Because the taps are spaced 2 apart,
 * the output stamps the 9 kernel values at every-other position -- you can SEE
 * the holes (the dilation pattern) in the output grid.
 *
 * Build:  gcc 13_dilated3x3_r2.c -o 13_dilated3x3_r2 -lm
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

void dilated3x3(const float *in, const float *w, float *out,
                int C, int H, int W, int rate)
{
    /* rate multiplies the kernel step to create gaps between taps:
       rate=1 -> standard 3x3 (3x3 footprint); rate=2 -> 5x5 footprint, 9 taps with holes */
    for (int c = 0; c < C; c++) {                  /* c: channel lane */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) {   /* kh: kernel row (0..2) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..2) */

                        /* (kh-1)*rate: center kernel at 0, offset by rate per step -> gaps between taps */
                        int ih = h + (kh - 1) * rate;    /* spaced-out taps: row offset = (kh-1)*rate */
                        int iw = x + (kw - 1) * rate;    /* spaced-out taps: col offset = (kw-1)*rate */
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            continue;              /* outside image boundary = zero padding */
                        }
                        /* input flat index:  [c][ih][iw] -> c*H*W + ih*W + iw    */
                        /* weight flat index: [c][kh][kw] -> c*K*K + kh*K + kw    */
                        sum += in[c*H*W + ih*W + iw] * w[c*K*K + kh*K + kw];
                    }
                }
                /* output flat index: [c][h][x] -> c*H*W + h*W + x */
                out[c*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 7, W = 7, rate = 2;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    in[3*W + 3] = 1.0f;                         /* impulse at the center */
    for (int i = 0; i < K*K; i++) {
        w[i] = (float)(i + 1);                  /* numbered kernel 1..9 */
    }

    show("INPUT  (impulse at center)", in, C, H, W);
    show("KERNEL (numbered 1..9)", w, C, K, K);
    dilated3x3(in, w, out, C, H, W, rate);
    show("OUTPUT (taps land 2 apart -- note the holes)", out, C, H, W);

    /* center still maps to the middle weight = 5 */
    printf("[13] check: center out[3][3] = %.1f (expected 5.0)\n", out[3*W + 3]);

    free(in);
    free(w);
    free(out);
    return 0;
}
