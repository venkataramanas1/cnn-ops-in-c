/* ============================================================================
 * [25] Residual Block  --  3x3 -> 3x3 + skip connection
 * ----------------------------------------------------------------------------
 * SHAPE: two 3x3 squares in sequence, with the ORIGINAL INPUT added back via a
 * wire that bypasses both. From the top you see two tic-tac-toe grids, then an
 * arrow that loops around and adds to the result. This is the key insight of
 * ResNet: the block only needs to learn the RESIDUAL (difference from identity).
 *
 *   Input : [C][H][W]
 *   Weight1: [C][C][3][3]
 *   Weight2: [C][C][3][3]
 *   Output : [C][H][W]   =  Conv2(ReLU(Conv1(x))) + x
 *
 * FLOPs ~ 2 * C * C * H * W * 9 * 2   (two 3x3 convs)
 *
 * DEMO: a constant feature map. After two all-ones conv (box blurs), the
 * skip adds back the original -- the output is larger than the conv output
 * alone. You can see the skip adding a baseline and the convs adding detail.
 *
 * Build:  gcc 25_residual_block.c -o 25_residual_block -lm
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

static void relu(float *x, int n)
{
    /* clamp negatives to zero in-place; identity on non-negative values */
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}

static void conv3x3(const float *in, const float *w, float *out,
                    int C, int H, int W)
{
    /* standard 3x3 same-spatial conv (PAD=1 keeps H x W unchanged);
       used twice in the residual block before the skip add */
    for (int co = 0; co < C; co++) { /* co: output channel */
        for (int h = 0; h < H; h++) { /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */

                float sum = 0.0f;
                for (int ci = 0; ci < C; ci++) { /* ci: input channel */
                    for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                            int ih = h + kh - PAD; /* ih: input row with padding offset */
                            int iw = x + kw - PAD; /* iw: input col with padding offset */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue; /* outside image boundary = zero padding */
                            }
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*C+ci)*K+kh)*K+kw */
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*C + ci)*K + kh)*K + kw];
                        }
                    }
                }
                out[co*H*W + h*W + x] = sum; /* output flat index: [co][h][x] */
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 4, W = 4;

    float *x   = calloc(C*H*W,   sizeof(float));
    float *w1  = calloc(C*C*K*K, sizeof(float));
    float *t1  = calloc(C*H*W,   sizeof(float));
    float *w2  = calloc(C*C*K*K, sizeof(float));
    float *t2  = calloc(C*H*W,   sizeof(float));
    float *out = calloc(C*H*W,   sizeof(float));

    /* input = constant 1.0 */
    for (int i = 0; i < C*H*W; i++) { x[i] = 1.0f; }
    /* small weights so conv output stays readable */
    for (int i = 0; i < C*C*K*K; i++) { w1[i] = 0.1f; w2[i] = 0.1f; }

    conv3x3(x,  w1, t1, C, H, W);
    relu(t1, C*H*W);
    conv3x3(t1, w2, t2, C, H, W);

    /* skip: output = conv2(relu(conv1(x))) + x */
    for (int i = 0; i < C*H*W; i++) {
        out[i] = t2[i] + x[i];
    }

    show("INPUT  x (constant 1.0)", x, C, H, W);
    show("After conv1+relu: t1 (box sum * 0.1)", t1, C, H, W);
    show("After conv2:      t2", t2, C, H, W);
    show("OUTPUT = t2 + x  (residual: skip brings back the baseline)", out, C, H, W);

    /* center: conv1 center = 9*0.1=0.9; conv2 center=9*0.9*0.1=0.81; +skip 1.0 */
    printf("[25] check: center out = %.4f (expected ~1.81)\n", out[1*W + 1]);

    free(x); free(w1); free(t1); free(w2); free(t2); free(out);
    return 0;
}
