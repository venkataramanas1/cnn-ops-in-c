/* ============================================================================
 * [07] Conv 1x1 -> 3x3 -> 1x1   (the bottleneck channel hourglass)
 * ----------------------------------------------------------------------------
 * SHAPE: dot -> square -> dot. Narrow, wide-spatial-work, narrow. The first 1x1
 * reduces channels so the expensive 3x3 runs cheap; the final 1x1 restores
 * width. This is the spatial core of a ResNet bottleneck (see [26] for the skip).
 *
 *   Input  : [C_in ][H][W]
 *   Weight1: [C_red][C_in ][1][1]        (1x1 reduce : C_in  -> C_red)
 *   Weight2: [C_red][C_red][3][3]        (3x3 spatial: C_red -> C_red)
 *   Weight3: [C_out][C_red][1][1]        (1x1 expand : C_red -> C_out)
 *   Output : [C_out][H][W]
 *
 * FLOPs ~ 2*H*W*( C_in*C_red + 9*C_red*C_red + C_red*C_out )
 *
 * DEMO: 2 input channels (impulses 1 and 1) -> reduce to 1 channel (spike 2) ->
 * 3x3 numbered stamp -> expand back to 2 channels. Narrow in the middle, wide
 * at the ends.
 *
 * Build:  gcc 07_bottleneck_1x1_3x3_1x1.c -o 07_bottleneck_1x1_3x3_1x1 -lm
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

static void conv1x1(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    /* step 1 (reduce) or step 3 (expand) — same 1x1 dot-product, caller sets Ci/Co */
    for (int co = 0; co < Co; co++) {          /* co: output channel */
        for (int h = 0; h < H; h++) {          /* h: output row */
            for (int x = 0; x < W; x++) {      /* x: output col */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {  /* ci: input channel being summed */
                    /* input flat index: [ci][h][x] -> ci*H*W + h*W + x */
                    /* weight flat index: [co][ci]  -> co*Ci + ci        */
                    sum += in[ci*H*W + h*W + x] * w[co*Ci + ci];
                }
                /* output flat index: [co][h][x] -> co*H*W + h*W + x */
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

static void conv3x3(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    const int K = 3, P = 1;
    /* step 2: spatial conv — 3x3 kernel slides over the reduced-channel feature map */
    for (int co = 0; co < Co; co++) {              /* co: output channel */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {      /* ci: input channel */
                    for (int kh = 0; kh < K; kh++) {   /* kh: kernel row (0..2) */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..2) */

                            int ih = h + kh - P;   /* input row: center output at h, offset by kh, undo padding P */
                            int iw = x + kw - P;   /* input col: same logic column-wise */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue;          /* outside image boundary = zero padding */
                            }
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw   */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*Ci+ci)*K+kh)*K+kw */
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*Ci + ci)*K + kh)*K + kw];
                        }
                    }
                }
                out[co*H*W + h*W + x] = sum;       /* output flat index: [co][h][x] */
            }
        }
    }
}

int main(void)
{
    int Cin = 2, Cred = 1, Cout = 2, H = 5, W = 5;

    float *in  = calloc(Cin*H*W,     sizeof(float));
    float *wa  = calloc(Cred*Cin,    sizeof(float));
    float *t1  = calloc(Cred*H*W,    sizeof(float));
    float *wb  = calloc(Cred*Cred*9, sizeof(float));
    float *t2  = calloc(Cred*H*W,    sizeof(float));
    float *wc  = calloc(Cout*Cred,   sizeof(float));
    float *out = calloc(Cout*H*W,    sizeof(float));

    in[0*H*W + 2*W + 2] = 1.0f;                 /* impulse in both input channels */
    in[1*H*W + 2*W + 2] = 1.0f;
    wa[0] = 1.0f; wa[1] = 1.0f;                 /* reduce: add the two channels */
    for (int i = 0; i < 9; i++) {
        wb[i] = (float)(i + 1);                 /* numbered 3x3 kernel */
    }
    wc[0] = 1.0f; wc[1] = 1.0f;                 /* expand: copy to both outputs */

    show("INPUT  (2 channels, impulses)", in, Cin, H, W);
    conv1x1(in, wa, t1, Cin, Cred, H, W);
    show("REDUCED (1 channel, spike 2)", t1, Cred, H, W);
    show("KERNEL 3x3 (numbered 1..9)", wb, 1, 3, 3);
    conv3x3(t1, wb, t2, Cred, Cred, H, W);
    show("SPATIAL (kernel stamped)", t2, Cred, H, W);
    conv1x1(t2, wc, out, Cred, Cout, H, W);
    show("EXPANDED (2 channels again)", out, Cout, H, W);

    /* center: spike 2 * middle kernel 5 = 10, copied to both output channels */
    printf("[07] check: out ch0 center = %.1f, ch1 center = %.1f (expected 10.0)\n",
           out[0*H*W + 2*W + 2], out[1*H*W + 2*W + 2]);

    free(in);
    free(wa);
    free(t1);
    free(wb);
    free(t2);
    free(wc);
    free(out);
    return 0;
}
