/* ============================================================================
 * [11] Depthwise Conv 5x5
 * ----------------------------------------------------------------------------
 * SHAPE: a 5x5 square PER channel. Like [08] but each isolated lane sees a
 * wider neighborhood. Still zero cross-channel mixing. Same code, K=5, P=2.
 *
 *   Input : [C][H][W]
 *   Weight: [C][5][5]                (one 5x5 per channel)
 *   Output: [C][H][W]                (C_out == C_in == C)
 *
 * FLOPs ~ 2 * C * H * W * 25
 *
 * DEMO: an impulse box-blurred by a 5x5 all-ones kernel spreads into a 5x5
 * block -- wider than the 3x3 blur of [08].
 *
 * Build:  gcc 11_depthwise5x5.c -o 11_depthwise5x5 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K 5
#define P 2

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

void depthwise5x5(const float *in, const float *w, float *out, int C, int H, int W)
{
    /* each channel is its own independent lane — wider 5x5 footprint, still no cross-channel mixing */
    for (int c = 0; c < C; c++) {                  /* c: channel lane (C_out == C_in) */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) {   /* kh: kernel row (0..4, K=5) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..4, K=5) */

                        int ih = h + kh - P;       /* input row: output row h, shifted by kh, minus padding P=2 */
                        int iw = x + kw - P;       /* input col: same logic column-wise */
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
    int C = 1, H = 7, W = 7;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    in[3*W + 3] = 1.0f;                         /* impulse at the center */
    for (int i = 0; i < C*K*K; i++) {
        w[i] = 1.0f;                            /* 5x5 box blur */
    }

    show("INPUT  (impulse)", in, C, H, W);
    show("KERNEL (all-ones 5x5)", w, C, K, K);
    depthwise5x5(in, w, out, C, H, W);
    show("OUTPUT (spread into a 5x5 block)", out, C, H, W);

    printf("[11] check: center out[3][3] = %.1f (expected 1.0)\n", out[3*W + 3]);

    free(in);
    free(w);
    free(out);
    return 0;
}
