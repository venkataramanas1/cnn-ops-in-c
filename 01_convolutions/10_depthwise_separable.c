/* ============================================================================
 * [10] Depthwise Separable Convolution  (MobileNet's core trick)
 * ----------------------------------------------------------------------------
 * SHAPE: a stack of independent 3x3 squares (one per channel) FOLLOWED BY a
 * single dot that mixes their outputs. Spatially it is "3x3 then 1x1", but the
 * 3x3s are isolated lanes. It factorizes a normal conv into two cheap steps:
 *      depthwise (spatial only)  +  pointwise (channel only)
 * Cost drops from  Ci*Co*9  to  Ci*9 + Ci*Co  -- often ~8-9x fewer MACs.
 *
 *   Input  : [C_in ][H][W]
 *   Weight1: [C_in ][3][3]               (depthwise: one 3x3 per channel)
 *   Weight2: [C_out][C_in][1][1]         (pointwise: mixes the lanes)
 *   Output : [C_out][H][W]
 *
 * FLOPs ~ 2*H*W*( 9*C_in  +  C_in*C_out )
 *
 * DEMO: 2 lanes, each box-blurs an impulse (spatial), then the pointwise sums
 * the two blurred lanes into one output channel (channel mix).
 *
 * Build:  gcc 10_depthwise_separable.c -o 10_depthwise_separable -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K 3
#define P 1

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

static void depthwise(const float *in, const float *w, float *out, int C, int H, int W)
{
    /* step 1: spatial filter — each channel is its own independent lane, no cross-channel mixing */
    for (int c = 0; c < C; c++) {                  /* c: channel lane (C_out == C_in) */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) {   /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                        int ih = h + kh - P;       /* input row: output row h, shifted by kh, minus padding P */
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

static void pointwise(const float *in, const float *w, float *out,
                      int Ci, int Co, int H, int W)
{
    /* step 2: channel mix — cross-wire the depthwise lanes into C_out output channels */
    for (int co = 0; co < Co; co++) {              /* co: output channel */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {  /* ci: input channel (depthwise lane) being mixed */
                    /* input flat index:  [ci][h][x] -> ci*H*W + h*W + x */
                    /* weight flat index: [co][ci]   -> co*Ci + ci        */
                    sum += in[ci*H*W + h*W + x] * w[co*Ci + ci];
                }
                /* output flat index: [co][h][x] -> co*H*W + h*W + x */
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C = 2, Co = 1, H = 5, W = 5;

    float *in  = calloc(C*H*W,  sizeof(float));
    float *wd  = calloc(C*K*K,  sizeof(float));
    float *mid = calloc(C*H*W,  sizeof(float));
    float *wp  = calloc(Co*C,   sizeof(float));
    float *out = calloc(Co*H*W, sizeof(float));

    in[0*H*W + 2*W + 2] = 1.0f;                 /* impulse in both lanes */
    in[1*H*W + 2*W + 2] = 1.0f;
    for (int i = 0; i < C*K*K; i++) {
        wd[i] = 1.0f;                           /* box blur each lane */
    }
    wp[0] = 1.0f; wp[1] = 1.0f;                 /* pointwise adds the lanes */

    show("INPUT  (2 lanes, impulses)", in, C, H, W);
    depthwise(in, wd, mid, C, H, W);
    show("AFTER DEPTHWISE (each lane box-blurred to a 3x3 block)", mid, C, H, W);
    pointwise(mid, wp, out, C, Co, H, W);
    show("AFTER POINTWISE (lanes summed into 1 channel)", out, Co, H, W);

    /* center: each lane = 1 there, summed = 2 */
    printf("[10] check: center out[2][2] = %.1f (expected 2.0)\n", out[2*W + 2]);

    free(in);
    free(wd);
    free(mid);
    free(wp);
    free(out);
    return 0;
}
