/* ============================================================================
 * [23] Grouped Conv g=C  --  the Depthwise extreme
 * ----------------------------------------------------------------------------
 * SHAPE: C independent 1-channel kernels -- one tiny patch per channel.
 * This is exactly depthwise convolution ([08]) arrived at from the grouped
 * convolution formulation. When g == C_in, each group has exactly one input
 * channel, so no channel mixing occurs at all. The extreme case of grouping.
 *
 *   Input : [C][H][W]
 *   Weight: [C][1][3][3]             (g = C groups, each 1 channel wide)
 *   Output: [C][H][W]                (C_out == C_in == C, no mixing)
 *
 * FLOPs ~ 2 * C * H * W * 9
 *
 * DEMO: 3 channels with different patterns (impulse, ramp, constant). Because
 * g=C, the 3x3 blur is applied separately to each -- you can verify the output
 * channels are fully independent (no cross-channel leakage).
 *
 * Build:  gcc 23_grouped_conv_gC_depthwise.c -o 23_grouped_conv_gC_depthwise -lm
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

/* grouped conv with g = C_in = C_out (= pure depthwise) */
void depthwise_via_groups(const float *in, const float *w, float *out,
                          int C, int H, int W)
{
    /* when g == C every channel is its own group of size 1:
       no channel mixing ever occurs -- each 3x3 kernel operates
       entirely within its own channel slice */
    for (int c = 0; c < C; c++) {                  /* each channel is its own group */
        for (int h = 0; h < H; h++) { /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                        int ih = h + kh - PAD; /* ih: input row with padding offset */
                        int iw = x + kw - PAD; /* iw: input col with padding offset */
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            continue; /* outside image boundary = zero padding */
                        }
                        /* input flat index: [c][ih][iw] -> c*H*W + ih*W + iw */
                        /* weight flat index: [c][kh][kw] -> c*K*K + kh*K + kw (one kernel per channel) */
                        sum += in[c*H*W + ih*W + iw] * w[c*K*K + kh*K + kw];
                    }
                }
                out[c*H*W + h*W + x] = sum; /* output flat index: [c][h][x] -> c*H*W + h*W + x */
            }
        }
    }
}

int main(void)
{
    int C = 3, H = 5, W = 5;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    /* channel 0: impulse at center */
    in[0*H*W + 2*W + 2] = 9.0f;
    /* channel 1: horizontal ramp (col index) */
    for (int h = 0; h < H; h++) {
        for (int x = 0; x < W; x++) {
            in[1*H*W + h*W + x] = (float)x;
        }
    }
    /* channel 2: constant = 5.0 */
    for (int i = 0; i < H*W; i++) {
        in[2*H*W + i] = 5.0f;
    }

    /* all-ones kernel for every channel */
    for (int i = 0; i < C*K*K; i++) {
        w[i] = 1.0f;
    }

    show("INPUT  (ch0=impulse, ch1=ramp, ch2=constant)", in, C, H, W);
    show("KERNEL (all-ones 3x3 per channel)", w, C, K, K);
    depthwise_via_groups(in, w, out, C, H, W);
    show("OUTPUT (each channel processed independently)", out, C, H, W);

    printf("[23] check: ch0 center = %.1f (expected 9.0)\n", out[0*H*W + 2*W + 2]);
    printf("[23] check: ch2 center = %.1f (expected 45.0 = 9*5)\n", out[2*H*W + 2*W + 2]);

    free(in); free(w); free(out);
    return 0;
}
