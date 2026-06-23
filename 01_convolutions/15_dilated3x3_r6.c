/* ============================================================================
 * [15] Dilated (atrous) Conv 3x3, rate = 6
 * ----------------------------------------------------------------------------
 * SHAPE: 9 points spread across a 13x13 area. From above: 9 widely-spaced dots
 * with a lot of empty space between them. The pattern is still a square
 * arrangement but the dots feel almost disconnected.
 *
 *   Input : [C][H][W]
 *   Weight: [C][3][3]                (9 weights, effective footprint 13x13)
 *   Output: [C][H][W]
 *
 * FLOPs ~ 2 * C * H * W * 9
 *
 * DEMO: same impulse + numbered-kernel demo as [13] and [14]. The output dots
 * are now 6 apart. Lay [13], [14], [15] side by side to see the field grow.
 *
 * Build:  gcc 15_dilated3x3_r6.c -o 15_dilated3x3_r6 -lm
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

void dilated3x3_r6(const float *in, const float *w, float *out,
                   int C, int H, int W, int rate)
{
    /* dilated conv rate=6: same gap formula (kh-1)*rate, but rate=6 pushes
     * the 9 sample points so far apart they feel almost disconnected;
     * effective receptive field = (K-1)*rate+1 = 13 wide */
    for (int c = 0; c < C; c++) {         /* c: output/input channel (depthwise) */
        for (int h = 0; h < H; h++) {     /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                        /* gap formula: same as all other dilated convs, only rate differs */
                        int ih = h + (kh - 1) * rate; /* input row: offset by (kh-1)*rate */
                        int iw = x + (kw - 1) * rate; /* input col: offset by (kw-1)*rate */
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
    int C = 1, H = 15, W = 15, rate = 6;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    in[7*W + 7] = 1.0f;                         /* impulse at center */
    for (int i = 0; i < K*K; i++) {
        w[i] = (float)(i + 1);                  /* numbered kernel 1..9 */
    }

    show("INPUT  (impulse at center)", in, C, H, W);
    show("KERNEL (numbered 1..9)", w, C, K, K);
    dilated3x3_r6(in, w, out, C, H, W, rate);
    show("OUTPUT (taps 6 cells apart -- almost disconnected)", out, C, H, W);

    printf("[15] check: center out[7][7] = %.1f (expected 5.0)\n", out[7*W + 7]);

    free(in);
    free(w);
    free(out);
    return 0;
}
