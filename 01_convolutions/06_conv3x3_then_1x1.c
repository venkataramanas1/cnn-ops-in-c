/* ============================================================================
 * [06] Conv 3x3 -> 1x1   (reverse of [05])
 * ----------------------------------------------------------------------------
 * SHAPE: square first, then a dot. Read the spatial neighborhood at full
 * channel width, THEN collapse/project channels with the 1x1. "Look then mix."
 *
 *   Input  : [C_in ][H][W]
 *   Weight1: [C_mid][C_in ][3][3]        (3x3 spatial:  C_in  -> C_mid)
 *   Weight2: [C_out][C_mid][1][1]        (1x1 project:  C_mid -> C_out)
 *   Output : [C_out][H][W]
 *
 * FLOPs ~ 2*H*W*( 9*C_in*C_mid  +  C_mid*C_out )
 *
 * DEMO: impulse -> the 3x3 numbered kernel stamps a spatial pattern -> the 1x1
 * then doubles every value. Spatial work happens FIRST, channel scaling LAST.
 *
 * Build:  gcc 06_conv3x3_then_1x1.c -o 06_conv3x3_then_1x1 -lm
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

/* 3x3 conv: looks at a 3x3 spatial neighborhood for every input channel */
static void conv3x3(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    const int K = 3, P = 1;
    for (int co = 0; co < Co; co++) {            /* co: output channel          */
        for (int h = 0; h < H; h++) {            /* h:  output row              */
            for (int x = 0; x < W; x++) {        /* x:  output column           */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {        /* ci: input channel   */
                    for (int kh = 0; kh < K; kh++) {     /* kh: kernel row      */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel column   */

                            int ih = h + kh - P;   /* input row after pad shift  */
                            int iw = x + kw - P;   /* input col after pad shift  */

                            /* outside the image boundary = zero (zero padding)  */
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

/* 1x1 conv: no spatial loop, just a channel dot-product at every pixel */
static void conv1x1(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    for (int co = 0; co < Co; co++) {            /* co: output channel */
        for (int h = 0; h < H; h++) {
            for (int x = 0; x < W; x++) {

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {   /* dot product over input channels */
                    /* weight [co][ci] is a flat 2D matrix -- no spatial dims */
                    sum += in[ci*H*W + h*W + x] * w[co*Ci + ci];
                }
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int Ci = 1, Cmid = 1, Co = 1, H = 5, W = 5;

    float *in  = calloc(Ci*H*W,    sizeof(float));
    float *w3  = calloc(Cmid*Ci*9, sizeof(float));
    float *mid = calloc(Cmid*H*W,  sizeof(float));  /* spatial result before scaling */
    float *w1  = calloc(Co*Cmid,   sizeof(float));
    float *out = calloc(Co*H*W,    sizeof(float));

    in[2*W + 2] = 1.0f;                         /* impulse */
    for (int i = 0; i < 9; i++) {
        w3[i] = (float)(i + 1);                 /* numbered 3x3 kernel */
    }
    w1[0] = 2.0f;                               /* 1x1 doubles the channel */

    show("INPUT  (impulse)", in, Ci, H, W);
    show("KERNEL1 3x3 (numbered 1..9)", w3, 1, 3, 3);

    /* step 1: 3x3 does all the spatial reading */
    conv3x3(in, w3, mid, Ci, Cmid, H, W);
    show("MID    (kernel stamped, flipped)", mid, Cmid, H, W);

    show("KERNEL2 1x1 (x2)", w1, 1, Co, Cmid);
    /* step 2: 1x1 rescales channels without touching space */
    conv1x1(mid, w1, out, Cmid, Co, H, W);
    show("OUTPUT (same pattern, doubled)", out, Co, H, W);

    /* center kernel weight 5, doubled = 10 */
    printf("[06] check: center out[2][2] = %.1f (expected 10.0)\n", out[2*W + 2]);

    free(in);
    free(w3);
    free(mid);
    free(w1);
    free(out);
    return 0;
}
