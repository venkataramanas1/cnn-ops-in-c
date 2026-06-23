/* ============================================================================
 * [26] Bottleneck Block  --  1x1 -> 3x3 -> 1x1 + skip (full ResNet block)
 * ----------------------------------------------------------------------------
 * SHAPE: dot -> square -> dot, with a bypass wire (skip). The spatial work
 * happens only in the middle 3x3 on a REDUCED channel count. The skip adds
 * back the original input. Bottleneck (narrow) refers to the channel hourglass,
 * not the spatial footprint. This is [07] with the residual connection added.
 *
 *   Input  : [C_in ][H][W]
 *   Weight1: [C_red][C_in ][1][1]     (1x1 reduce)
 *   Weight2: [C_red][C_red][3][3]     (3x3 spatial)
 *   Weight3: [C_out][C_red][1][1]     (1x1 expand)
 *   Output : [C_out][H][W]  =  expand(...) + x   (C_out == C_in for identity skip)
 *
 * FLOPs ~ 2*H*W*(C_in*C_red + 9*C_red^2 + C_red*C_out) + C_out*H*W (add)
 *
 * DEMO: impulse input, the three convs process it, then the skip brings back
 * the original impulse. Watch the output = processed + original.
 *
 * Build:  gcc 26_bottleneck_block_with_skip.c -o 26_bottleneck_block_with_skip -lm
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
                printf("%6.2f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void relu(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) { x[i] = 0.0f; }
    }
}
static void conv1x1(const float *in, const float *w, float *out, int Ci, int Co, int H, int W)
{
    /* 1x1 conv = per-pixel channel projection; no spatial footprint,
       used for the reduce (Ci->Cred) and expand (Cred->Co) steps */
    for (int co = 0; co < Co; co++) { /* co: output channel */
        for (int h = 0; h < H; h++) { /* h: spatial row */
            for (int x = 0; x < W; x++) { /* x: spatial col */
                float s = 0.0f;
                for (int ci = 0; ci < Ci; ci++) { /* ci: input channel */
                    /* weight flat index: [co][ci] -> co*Ci + ci (no spatial dims) */
                    /* input flat index: [ci][h][x] -> ci*H*W + h*W + x */
                    s += in[ci*H*W+h*W+x] * w[co*Ci+ci];
                }
                out[co*H*W+h*W+x] = s; /* output flat index: [co][h][x] */
            }
        }
    }
}
static void conv3x3(const float *in, const float *w, float *out, int Ci, int Co, int H, int W)
{
    /* 3x3 same-spatial conv (PAD=1); the spatial mixing step in the
       bottleneck -- operates on the reduced (narrow) channel count */
    for (int co = 0; co < Co; co++) { /* co: output channel */
        for (int h = 0; h < H; h++) { /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */
                float s = 0.0f;
                for (int ci = 0; ci < Ci; ci++) { /* ci: input channel */
                    for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */
                            int ih = h+kh-PAD, iw = x+kw-PAD; /* ih/iw: input coords with padding offset */
                            if (ih<0||ih>=H||iw<0||iw>=W) { continue; } /* outside image boundary = zero padding */
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*Ci+ci)*K+kh)*K+kw */
                            s += in[ci*H*W+ih*W+iw]*w[((co*Ci+ci)*K+kh)*K+kw];
                        }
                    }
                }
                out[co*H*W+h*W+x] = s; /* output flat index: [co][h][x] */
            }
        }
    }
}

int main(void)
{
    int Cin = 2, Cred = 1, Cout = 2, H = 5, W = 5;

    float *x   = calloc(Cin *H*W,    sizeof(float));
    float *wa  = calloc(Cred*Cin,    sizeof(float));
    float *t1  = calloc(Cred*H*W,    sizeof(float));
    float *wb  = calloc(Cred*Cred*9, sizeof(float));
    float *t2  = calloc(Cred*H*W,    sizeof(float));
    float *wc  = calloc(Cout*Cred,   sizeof(float));
    float *t3  = calloc(Cout*H*W,    sizeof(float));
    float *out = calloc(Cout*H*W,    sizeof(float));

    x[0*H*W + 2*W + 2] = 4.0f;             /* impulse in ch0 */
    x[1*H*W + 2*W + 2] = 4.0f;             /* impulse in ch1 */

    wa[0]=1.0f; wa[1]=1.0f;                 /* reduce: sum channels */
    for (int i=0;i<9;i++) { wb[i]=1.0f; }  /* 3x3 box blur */
    wc[0]=0.5f; wc[1]=0.5f;                 /* expand: split evenly */

    conv1x1(x,  wa, t1, Cin,  Cred, H, W);
    relu(t1, Cred*H*W);
    conv3x3(t1, wb, t2, Cred, Cred, H, W);
    relu(t2, Cred*H*W);
    conv1x1(t2, wc, t3, Cred, Cout, H, W);

    /* skip: out = t3 + x */
    for (int i = 0; i < Cout*H*W; i++) {
        out[i] = t3[i] + x[i];
    }

    show("INPUT  x (impulse 4 in both channels)", x, Cin, H, W);
    show("BOTTLENECK output t3 (before skip)", t3, Cout, H, W);
    show("OUTPUT = t3 + x  (skip restores the impulse baseline)", out, Cout, H, W);

    /* reduce: 4+4=8; box blur center=8; expand *0.5=4; skip +4 = 8 */
    printf("[26] check: out ch0 center = %.2f (expected 8.00)\n",
           out[0*H*W + 2*W + 2]);

    free(x);free(wa);free(t1);free(wb);free(t2);free(wc);free(t3);free(out);
    return 0;
}
