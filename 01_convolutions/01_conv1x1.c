/* ============================================================================
 * [01] Conv 1x1  --  Pointwise Convolution
 * ----------------------------------------------------------------------------
 * SHAPE FROM THE TOP:  A single dot.  One pixel, no neighbors touched.
 *
 * It is a per-pixel weighted sum ACROSS channels at exactly one (h,w) location.
 * There is NO spatial kernel loop (no kh/kw) because the kernel is 1x1. All the
 * work happens in the channel dimension -- it mixes / projects channels.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][1][1]   (shown as a C_out x C_in mixing matrix)
 *   Bias  : [C_out]
 *   Output: [C_out][H][W]
 *
 * FLOPs ~ 2 * C_in * C_out * H * W      (a dot product of length C_in per pixel)
 *
 * DEMO: 3 input channels each a flat constant (1, 2, 3). The 1x1 does NOT move
 * in space -- it only re-mixes channels, so every pixel of an output channel is
 * identical. Output channel 0 = 1*1 + 1*2 + 1*3 = 6 everywhere (a channel sum).
 *
 * Build:  gcc 01_conv1x1.c -o 01_conv1x1 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

/* ---- tiny terminal visualizer: prints each channel as a clean number grid -- */
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

void conv1x1(const float *in, const float *w, const float *bias, float *out,
             int C_in, int C_out, int H, int W)
{
    for (int co = 0; co < C_out; co++) {         /* co: which output channel we are filling */
        for (int h = 0; h < H; h++) {            /* h:  row position in the spatial map     */
            for (int x = 0; x < W; x++) {        /* x:  column position in the spatial map  */

                /* start the accumulator at bias (or 0 if no bias provided) */
                float sum = bias ? bias[co] : 0.0f;

                for (int ci = 0; ci < C_in; ci++) {   /* ci: input channel being dotted */

                    /* input  flat index:  [ci][h][x]  ->  ci*H*W + h*W + x             */
                    /* weight flat index:  [co][ci]    ->  co*C_in + ci  (1x1 = no kh,kw) */
                    sum += in[ci*H*W + h*W + x] * w[co*C_in + ci];
                }

                /* write result: output flat index [co][h][x] -> co*H*W + h*W + x */
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C_in = 3, C_out = 2, H = 3, W = 3;

    float *in  = calloc(C_in *H*W,  sizeof(float));
    float *w   = calloc(C_out*C_in, sizeof(float));
    float *out = calloc(C_out*H*W,  sizeof(float));

    /* each input channel is a flat constant: ch0=1, ch1=2, ch2=3 */
    for (int c = 0; c < C_in; c++) {
        for (int i = 0; i < H*W; i++) {
            in[c*H*W + i] = (float)(c + 1);
        }
    }
    /* out0 = sum of channels (1,1,1);  out1 = ch2 minus ch0 (-1,0,1) */
    w[0*C_in + 0] =  1;  w[0*C_in + 1] = 1;  w[0*C_in + 2] = 1;
    w[1*C_in + 0] = -1;  w[1*C_in + 1] = 0;  w[1*C_in + 2] = 1;

    show("INPUT  (ch0=1, ch1=2, ch2=3)", in, C_in, H, W);
    show("KERNEL (C_out x C_in mixing matrix)", w, 1, C_out, C_in);
    conv1x1(in, w, NULL, out, C_in, C_out, H, W);
    show("OUTPUT (ch0 = sum = 6;  ch1 = ch2-ch0 = 2)", out, C_out, H, W);

    printf("[01] check: out0 = %.1f (expected 6.0), out1 = %.1f (expected 2.0)\n",
           out[0], out[1*H*W]);

    free(in);
    free(w);
    free(out);
    return 0;
}
