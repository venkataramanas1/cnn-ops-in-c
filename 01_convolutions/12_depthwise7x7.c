/* ============================================================================
 * [12] Depthwise Conv 7x7   (ConvNeXt's signature layer)
 * ----------------------------------------------------------------------------
 * SHAPE: a broad 7x7 window PER channel, no cross-channel mixing at this stage.
 * ConvNeXt showed that a big depthwise kernel (cheap, because it is per-channel)
 * followed by pointwise 1x1s can rival attention. Same code, K=7, P=3.
 *
 *   Input : [C][H][W]
 *   Weight: [C][7][7]                (one 7x7 per channel)
 *   Output: [C][H][W]                (C_out == C_in == C)
 *
 * FLOPs ~ 2 * C * H * W * 49
 *
 * DEMO: an impulse box-blurred by a 7x7 all-ones kernel spreads into a 7x7
 * block -- the widest single-layer footprint in this folder.
 *
 * Build:  gcc 12_depthwise7x7.c -o 12_depthwise7x7 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K 7
#define P 3

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

void depthwise7x7(const float *in, const float *w, float *out, int C, int H, int W)
{
    /* each channel is its own independent lane — broad 7x7 footprint (ConvNeXt), still no cross-channel mixing */
    for (int c = 0; c < C; c++) {                  /* c: channel lane (C_out == C_in) */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) {   /* kh: kernel row (0..6, K=7) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..6, K=7) */

                        int ih = h + kh - P;       /* input row: output row h, shifted by kh, minus padding P=3 */
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
    int C = 1, H = 9, W = 9;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    in[4*W + 4] = 1.0f;                         /* impulse at the center */
    for (int i = 0; i < C*K*K; i++) {
        w[i] = 1.0f;                            /* 7x7 box blur */
    }

    show("INPUT  (impulse)", in, C, H, W);
    show("KERNEL (all-ones 7x7)", w, C, K, K);
    depthwise7x7(in, w, out, C, H, W);
    show("OUTPUT (spread into a 7x7 block)", out, C, H, W);

    printf("[12] check: center out[4][4] = %.1f (expected 1.0)\n", out[4*W + 4]);

    free(in);
    free(w);
    free(out);
    return 0;
}
