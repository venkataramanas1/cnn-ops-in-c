/* ============================================================================
 * [02] Conv 3x3  --  the workhorse of CNNs
 * ----------------------------------------------------------------------------
 * SHAPE FROM THE TOP: a tic-tac-toe grid (9 cells). The center is the anchor
 * pixel; the 8 surrounding cells are its immediate neighbors. It sees left,
 * right, up, down and all four diagonals at once -- the most natural spatial
 * receptive field.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][3][3]
 *   Bias  : [C_out]
 *   Output: [C_out][H][W]            (SAME padding = 1, stride 1 -> size kept)
 *
 * FLOPs ~ 2 * C_in * C_out * H * W * 9     (9 = 3x3 taps per channel pair)
 *
 * DEMO: feed a single bright impulse and a NUMBERED kernel (1..9). Where the
 * impulse sits, the output paints the kernel back -- but FLIPPED 180 degrees,
 * because convolution mirrors the kernel.
 *
 * Build:  gcc 02_conv3x3.c -o 02_conv3x3 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K   3
#define PAD 1

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

void conv3x3(const float *in, const float *w, const float *bias, float *out,
             int C_in, int C_out, int H, int W)
{
    for (int co = 0; co < C_out; co++) {         /* co: output channel index            */
        for (int h = 0; h < H; h++) {            /* h:  output row                      */
            for (int x = 0; x < W; x++) {        /* x:  output column                   */

                float sum = bias ? bias[co] : 0.0f;  /* bias initialises the accumulator */

                for (int ci = 0; ci < C_in; ci++) {      /* ci: input channel            */
                    for (int kh = 0; kh < K; kh++) {     /* kh: kernel row offset 0..2   */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel column offset 0..2 */

                            /* map output position + kernel offset back to input coords   */
                            /* PAD=1 keeps the output the same size as the input          */
                            int ih = h + kh - PAD;   /* input row    (may go negative -> padding) */
                            int iw = x + kw - PAD;   /* input column (may go negative -> padding) */

                            /* pixels that fall outside the image boundary are zero (zero pad) */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue;
                            }

                            /* input  flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw           */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*C_in+ci)*K+kh)*K+kw */
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*C_in + ci)*K + kh)*K + kw];
                        }
                    }
                }

                /* store result at output flat index [co][h][x] */
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C_in = 1, C_out = 1, H = 5, W = 5;

    float *in  = calloc(C_in*H*W,        sizeof(float));
    float *w   = calloc(C_out*C_in*K*K,  sizeof(float));
    float *out = calloc(C_out*H*W,       sizeof(float));

    in[2*W + 2] = 1.0f;                         /* single impulse at the center */
    for (int i = 0; i < K*K; i++) {
        w[i] = (float)(i + 1);                  /* numbered kernel 1..9 */
    }

    show("INPUT  (impulse at center)", in, C_in, H, W);
    show("KERNEL (numbered 1..9)", w, 1, K, K);
    conv3x3(in, w, NULL, out, C_in, C_out, H, W);
    show("OUTPUT (kernel painted back, flipped 180 deg)", out, C_out, H, W);

    printf("[02] check: center out[2][2] = %.1f (expected 5.0 = kernel middle)\n",
           out[2*W + 2]);

    free(in);
    free(w);
    free(out);
    return 0;
}
