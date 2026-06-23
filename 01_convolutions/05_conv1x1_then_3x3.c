/* ============================================================================
 * [05] Conv 1x1 -> 3x3
 * ----------------------------------------------------------------------------
 * SHAPE: a dot, then a tic-tac-toe square. The 1x1 first COMPRESSES channels
 * (cheap, no spatial awareness), then the 3x3 does the spatial reading on the
 * smaller channel count. "Reduce then look" (Inception-style branches).
 *
 *   Input  : [C_in ][H][W]
 *   Weight1: [C_mid][C_in ][1][1]        (1x1 squeeze:  C_in  -> C_mid)
 *   Weight2: [C_out][C_mid][3][3]        (3x3 spatial:  C_mid -> C_out)
 *   Output : [C_out][H][W]
 *
 * FLOPs ~ 2*H*W*( C_in*C_mid  +  9*C_mid*C_out )
 *
 * DEMO: 2 input channels (impulses of height 1 and 2). The 1x1 ADDS them into
 * one channel (a spike of 3). The 3x3 numbered kernel then stamps itself around
 * that spike. You watch two channels collapse to one, then spread in space.
 *
 * Build:  gcc 05_conv1x1_then_3x3.c -o 05_conv1x1_then_3x3 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

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

/* 1x1 conv: dot product over input channels, NO spatial loop */
static void conv1x1(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    for (int co = 0; co < Co; co++) {            /* co: output channel         */
        for (int h = 0; h < H; h++) {            /* h: row (unchanged by 1x1)  */
            for (int x = 0; x < W; x++) {        /* x: col (unchanged by 1x1)  */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {    /* dot product over Ci channels */
                    /* weight is a flat matrix [co][ci] -- no spatial dimension */
                    sum += in[ci*H*W + h*W + x] * w[co*Ci + ci];
                }
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

/* 3x3 conv: dot product over input channels AND 3x3 spatial neighborhood */
static void conv3x3(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    const int K = 3, P = 1;   /* K=kernel size, P=padding to keep size */
    for (int co = 0; co < Co; co++) {            /* co: output channel         */
        for (int h = 0; h < H; h++) {            /* h:  output row             */
            for (int x = 0; x < W; x++) {        /* x:  output column          */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {        /* ci: input channel   */
                    for (int kh = 0; kh < K; kh++) {     /* kh: kernel row 0..2 */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col 0..2 */

                            int ih = h + kh - P;   /* map kernel tap -> input row    */
                            int iw = x + kw - P;   /* map kernel tap -> input column */

                            /* skip taps that land outside the image (zero pad) */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue;
                            }
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*Ci + ci)*K + kh)*K + kw];
                        }
                    }
                }
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int Ci = 2, Cmid = 1, Co = 1, H = 5, W = 5;

    float *in  = calloc(Ci*H*W,   sizeof(float));
    float *w1  = calloc(Cmid*Ci,  sizeof(float));
    float *mid = calloc(Cmid*H*W, sizeof(float));  /* intermediate after 1x1 */
    float *w3  = calloc(Co*Cmid*9,sizeof(float));
    float *out = calloc(Co*H*W,   sizeof(float));

    in[0*H*W + 2*W + 2] = 1.0f;                 /* channel 0 impulse, height 1 */
    in[1*H*W + 2*W + 2] = 2.0f;                 /* channel 1 impulse, height 2 */
    w1[0] = 1.0f; w1[1] = 1.0f;                 /* 1x1 adds the two channels   */
    for (int i = 0; i < 9; i++) {
        w3[i] = (float)(i + 1);                 /* numbered 3x3 kernel         */
    }

    show("INPUT  (2 channels: impulses 1 and 2)", in, Ci, H, W);
    show("KERNEL1 1x1 (adds channels)", w1, 1, Cmid, Ci);

    /* step 1: 1x1 reduces Ci=2 channels to Cmid=1 (just sums them here) */
    conv1x1(in, w1, mid, Ci, Cmid, H, W);
    show("MID    (1 channel: spike of 3)", mid, Cmid, H, W);

    show("KERNEL2 3x3 (numbered 1..9)", w3, 1, 3, 3);
    /* step 2: 3x3 reads spatial neighbors of the compressed map */
    conv3x3(mid, w3, out, Cmid, Co, H, W);
    show("OUTPUT (kernel stamped around the merged spike)", out, Co, H, W);

    /* spike 3 * middle kernel weight 5 = 15 at the center */
    printf("[05] check: center out[2][2] = %.1f (expected 15.0)\n", out[2*W + 2]);

    free(in);
    free(w1);
    free(mid);
    free(w3);
    free(out);
    return 0;
}
