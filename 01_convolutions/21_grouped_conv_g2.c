/* ============================================================================
 * [21] Grouped Convolution g=2
 * ----------------------------------------------------------------------------
 * SHAPE: two side-by-side 3x3 squares, each operating on HALF the input
 * channels independently. Two parallel lanes that never see each other's inputs.
 * AlexNet introduced this to split across two GPUs; ShuffleNet uses it with
 * channel shuffle to exchange information between groups.
 *
 *   Input : [C_in ][H][W]         (split into g=2 groups of C_in/2 each)
 *   Weight: [C_out][C_in/g][3][3] (each output group has its own kernel set)
 *   Output: [C_out][H][W]
 *
 * FLOPs ~ 2 * C_in * C_out * H * W * 9 / g   (g times cheaper than dense)
 *
 * DEMO: 4 input channels split into 2 groups (ch0-1 vs ch2-3). Group 0 sees
 * only ch0+ch1; group 1 sees only ch2+ch3. The outputs show completely
 * independent processing of each half.
 *
 * Build:  gcc 21_grouped_conv_g2.c -o 21_grouped_conv_g2 -lm
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

void grouped_conv_g2(const float *in, const float *w, float *out,
                     int C_in, int C_out, int H, int W, int g)
{
    /* each group is an independent conv lane: group grp sees only
       input channels [ci_start .. ci_start+ci_per_g) and writes to
       output channels [co_start .. co_start+co_per_g) */
    int ci_per_g = C_in  / g; /* input channels per group */
    int co_per_g = C_out / g; /* output channels per group */

    for (int grp = 0; grp < g; grp++) { /* grp: group index (0..g-1) */

        int ci_start = grp * ci_per_g; /* first input channel for this group */
        int co_start = grp * co_per_g; /* first output channel for this group */

        for (int co = co_start; co < co_start + co_per_g; co++) { /* co: output channel (global) */
            for (int h = 0; h < H; h++) { /* h: output row */
                for (int x = 0; x < W; x++) { /* x: output col */

                    float sum = 0.0f;
                    for (int ci = ci_start; ci < ci_start + ci_per_g; ci++) { /* ci: input channel (group-local range) */
                        for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                            for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                                int ih = h + kh - PAD; /* ih: input row with padding offset */
                                int iw = x + kw - PAD; /* iw: input col with padding offset */
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue; /* outside image boundary = zero padding */
                                }
                                /* weight offset within group */
                                int local_co = co - co_start; /* local_co: output channel index within group */
                                int local_ci = ci - ci_start; /* local_ci: input channel index within group */
                                /* weight flat index: [grp*co_per_g*ci_per_g + local_co*ci_per_g + local_ci][kh][kw] */
                                int wi = ((co_start * ci_per_g + local_co * ci_per_g + local_ci) * K + kh) * K + kw;
                                sum += in[ci*H*W + ih*W + iw] * w[wi];
                            }
                        }
                    }
                    out[co*H*W + h*W + x] = sum; /* output flat index: [co][h][x] -> co*H*W + h*W + x */
                }
            }
        }
    }
}

int main(void)
{
    int C_in = 4, C_out = 4, H = 4, W = 4, g = 2;

    float *in  = calloc(C_in *H*W,          sizeof(float));
    float *w   = calloc(C_out*C_in/g*K*K,   sizeof(float));
    float *out = calloc(C_out*H*W,           sizeof(float));

    /* ch0=1, ch1=1, ch2=10, ch3=10 -- groups clearly different */
    for (int i = 0; i < H*W; i++) { in[0*H*W+i]=1.0f; in[1*H*W+i]=1.0f; }
    for (int i = 0; i < H*W; i++) { in[2*H*W+i]=10.0f; in[3*H*W+i]=10.0f; }
    for (int i = 0; i < C_out*C_in/g*K*K; i++) {
        w[i] = 1.0f;
    }

    show("INPUT  (ch0-1=1, ch2-3=10)", in, C_in, H, W);
    grouped_conv_g2(in, w, out, C_in, C_out, H, W, g);
    show("OUTPUT (ch0-1 from group0 = ~18; ch2-3 from group1 = ~180)", out, C_out, H, W);

    /* center of group0 output = 9 taps * 2 channels * 1 = 18 */
    printf("[21] check: out ch0 center = %.1f (expected 18.0)\n", out[0*H*W + 1*W + 1]);
    /* center of group1 output = 9 * 2 * 10 = 180 */
    printf("[21] check: out ch2 center = %.1f (expected 180.0)\n", out[2*H*W + 1*W + 1]);

    free(in); free(w); free(out);
    return 0;
}
