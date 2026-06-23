/* ============================================================================
 * [09] Pointwise Conv 1x1
 * ----------------------------------------------------------------------------
 * SHAPE: a single dot -- same footprint as [01]. The difference is PURPOSE:
 * here the 1x1 exists to CROSS-WIRE the outputs of depthwise lanes. Minimal
 * spatial shape; all the value is in channel mixing. It is the second half of
 * a depthwise-separable convolution.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][1][1]      (a C_out x C_in mixing matrix)
 *   Bias  : [C_out]
 *   Output: [C_out][H][W]
 *
 * FLOPs ~ 2 * C_in * C_out * H * W     (a dot product of length C_in per pixel)
 *
 * DEMO: 4 depthwise lanes, each a flat constant (1,2,3,4). One output channel
 * AVERAGES them (all weights 0.25) -> 2.5 everywhere. Pure channel cross-wiring.
 *
 * Build:  gcc 09_pointwise1x1.c -o 09_pointwise1x1 -lm
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
                printf("%6.2f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

void pointwise(const float *in, const float *w, const float *b, float *out,
               int Ci, int Co, int H, int W)
{
    /* purpose: cross-wire depthwise lanes — 1x1 spatial footprint, all value in channel mix */
    for (int co = 0; co < Co; co++) {              /* co: output channel */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = b ? b[co] : 0.0f;      /* optional per-output-channel bias */
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
    int Ci = 4, Co = 1, H = 3, W = 3;

    float *in  = calloc(Ci*H*W, sizeof(float));
    float *w   = calloc(Co*Ci,  sizeof(float));
    float *out = calloc(Co*H*W, sizeof(float));

    for (int c = 0; c < Ci; c++) {              /* lanes = 1,2,3,4 (flat) */
        for (int i = 0; i < H*W; i++) {
            in[c*H*W + i] = (float)(c + 1);
        }
    }
    for (int i = 0; i < Co*Ci; i++) {
        w[i] = 0.25f;                           /* average the 4 lanes */
    }

    show("INPUT  (4 lanes: 1,2,3,4)", in, Ci, H, W);
    show("KERNEL (1x4 averaging mix)", w, 1, Co, Ci);
    pointwise(in, w, NULL, out, Ci, Co, H, W);
    show("OUTPUT (their average = 2.5)", out, Co, H, W);

    printf("[09] check: out[0] = %.2f (expected 2.50)\n", out[0]);

    free(in);
    free(w);
    free(out);
    return 0;
}
