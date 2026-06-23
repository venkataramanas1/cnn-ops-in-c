/* ============================================================================
 * [22] Grouped Convolution g=4
 * ----------------------------------------------------------------------------
 * SHAPE: four parallel 3x3 squares, each covering a QUARTER of the channels.
 * Like [21] but with four independent workers. More parameter savings, less
 * cross-channel information flow (needs channel shuffle to compensate).
 *
 *   Input : [C_in ][H][W]         (split into 4 groups of C_in/4 each)
 *   Weight: [C_out][C_in/4][3][3]
 *   Output: [C_out][H][W]
 *
 * FLOPs ~ 2 * C_in * C_out * H * W * 9 / 4
 *
 * DEMO: 8 channels with values 1..8 split into 4 groups of 2. Print each
 * output group separately to see that group i only sees channels 2i and 2i+1.
 *
 * Build:  gcc 22_grouped_conv_g4.c -o 22_grouped_conv_g4 -lm
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
                printf("%7.1f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

void grouped_conv(const float *in, const float *w, float *out,
                  int C_in, int C_out, int H, int W, int g)
{
    /* g independent conv lanes; group grp handles input channels
       [grp*ci_pg .. (grp+1)*ci_pg) and output channels [grp*co_pg .. (grp+1)*co_pg).
       Weights are laid out contiguously per group in w[]. */
    int ci_pg = C_in  / g; /* input channels per group */
    int co_pg = C_out / g; /* output channels per group */

    for (int grp = 0; grp < g; grp++) { /* grp: group index (0..g-1) */
        for (int lco = 0; lco < co_pg; lco++) { /* lco: local output channel within group */
            int co = grp * co_pg + lco; /* co: global output channel index */
            for (int h = 0; h < H; h++) { /* h: output row */
                for (int x = 0; x < W; x++) { /* x: output col */

                    float sum = 0.0f;
                    for (int lci = 0; lci < ci_pg; lci++) { /* lci: local input channel within group */
                        int ci = grp * ci_pg + lci; /* ci: global input channel index */
                        for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                            for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                                int ih = h + kh - PAD; /* ih: input row with padding offset */
                                int iw = x + kw - PAD; /* iw: input col with padding offset */
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue; /* outside image boundary = zero padding */
                                }
                                /* weight flat index: [grp*co_pg*ci_pg + lco*ci_pg + lci][kh][kw] */
                                int wi = ((grp * co_pg * ci_pg + lco * ci_pg + lci) * K + kh) * K + kw;
                                /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
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
    int C_in = 8, C_out = 8, H = 4, W = 4, g = 4;

    float *in  = calloc(C_in *H*W,          sizeof(float));
    float *w   = calloc(C_out*C_in/g*K*K,   sizeof(float));
    float *out = calloc(C_out*H*W,           sizeof(float));

    /* channel c = constant value (c+1)*10 */
    for (int c = 0; c < C_in; c++) {
        for (int i = 0; i < H*W; i++) {
            in[c*H*W + i] = (float)((c + 1) * 10);
        }
    }
    for (int i = 0; i < C_out*C_in/g*K*K; i++) {
        w[i] = 1.0f;
    }

    show("INPUT  (ch0=10, ch1=20, ch2=30, ch3=40, ...)", in, C_in, H, W);
    grouped_conv(in, w, out, C_in, C_out, H, W, g);
    show("OUTPUT (each pair of output channels from its own group)", out, C_out, H, W);

    /* group0 sees ch0(10)+ch1(20)=30 per tap, 9 taps -> center = 270 */
    printf("[22] check: grp0 out ch0 center = %.1f (expected 270.0)\n",
           out[0*H*W + 1*W + 1]);

    free(in); free(w); free(out);
    return 0;
}
