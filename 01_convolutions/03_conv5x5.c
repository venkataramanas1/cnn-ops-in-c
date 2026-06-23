/* ============================================================================
 * [03] Conv 5x5  --  one extra ring of neighbors beyond 3x3 (25 cells)
 * ----------------------------------------------------------------------------
 * SHAPE FROM THE TOP: the 3x3 grid with ONE more ring wrapped around it. You
 * now see two pixels in every direction. Wider context, ~2.8x the MACs of 3x3.
 * Same code as conv3x3 but K=5, PAD=2.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][5][5]
 *   Bias  : [C_out]
 *   Output: [C_out][H][W]            (SAME padding = 2, stride 1 -> size kept)
 *
 * FLOPs ~ 2 * C_in * C_out * H * W * 25    (25 = 5x5 taps per channel pair)
 *
 * DEMO: impulse + numbered kernel 1..25. The output paints the whole 5x5 kernel
 * (flipped) around the impulse -- a wider stamp than the 3x3 case.
 *
 * Build:  gcc 03_conv5x5.c -o 03_conv5x5 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K   5
#define PAD 2     /* PAD = (K-1)/2 keeps output size equal to input size */

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

void conv5x5(const float *in, const float *w, const float *bias, float *out,
             int C_in, int C_out, int H, int W)
{
    for (int co = 0; co < C_out; co++) {         /* co: output channel */
        for (int h = 0; h < H; h++) {            /* h:  output row     */
            for (int x = 0; x < W; x++) {        /* x:  output column  */

                float sum = bias ? bias[co] : 0.0f;

                for (int ci = 0; ci < C_in; ci++) {       /* ci: input channel  */
                    for (int kh = 0; kh < K; kh++) {      /* kh: kernel row     */
                        for (int kw = 0; kw < K; kw++) {  /* kw: kernel column  */

                            /* PAD=2 means we look 2 pixels to the left/above center */
                            int ih = h + kh - PAD;   /* input row:    can be -2 to H+1 */
                            int iw = x + kw - PAD;   /* input column: can be -2 to W+1 */

                            /* clamp: anything outside [0,H) or [0,W) is treated as 0 */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue;
                            }

                            /* weight flat index: [co][ci][kh][kw] */
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*C_in + ci)*K + kh)*K + kw];
                        }
                    }
                }

                out[co*H*W + h*W + x] = sum;     /* output flat index: [co][h][x] */
            }
        }
    }
}

int main(void)
{
    int C_in = 1, C_out = 1, H = 7, W = 7;

    float *in  = calloc(C_in*H*W,       sizeof(float));
    float *w   = calloc(C_out*C_in*K*K, sizeof(float));
    float *out = calloc(C_out*H*W,      sizeof(float));

    in[3*W + 3] = 1.0f;                         /* impulse at the center */
    for (int i = 0; i < K*K; i++) {
        w[i] = (float)(i + 1);                  /* numbered kernel 1..25 */
    }

    show("INPUT  (impulse at center)", in, C_in, H, W);
    show("KERNEL (numbered 1..25)", w, 1, K, K);
    conv5x5(in, w, NULL, out, C_in, C_out, H, W);
    show("OUTPUT (5x5 kernel stamped, flipped)", out, C_out, H, W);

    /* center maps to the middle kernel weight = 13 */
    printf("[03] check: center out[3][3] = %.1f (expected 13.0)\n", out[3*W + 3]);

    free(in);
    free(w);
    free(out);
    return 0;
}
