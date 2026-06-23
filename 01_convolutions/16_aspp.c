/* ============================================================================
 * [16] ASPP Block -- Atrous Spatial Pyramid Pooling
 * ----------------------------------------------------------------------------
 * SHAPE: NOT one shape -- a FAMILY of four shapes applied simultaneously:
 *   - one 1x1 dot (no spatial awareness)
 *   - three dilated 3x3 grids (rates 2, 4, 6) getting progressively sparser
 *   - one global average pool that collapses everything to a single value
 * All five run in parallel, then their outputs are concatenated and projected.
 * Used in DeepLab for multi-scale context.
 *
 *   Input : [C_in][H][W]
 *   Weights: five separate kernels (1x1, three 3x3, 1 pool)
 *   Output : [5*C_branch][H][W]  (before final 1x1 fusion projection)
 *
 * FLOPs ~ sum of each branch's FLOPs
 *
 * DEMO: a diagonal gradient input is processed by all five branches in parallel.
 * Print each branch output separately so you can see how different dilation
 * rates capture the same scene at different spatial scales.
 *
 * Build:  gcc 16_aspp.c -o 16_aspp -lm
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

/* 1x1 conv -- identity projection (weight = 1.0 per channel) */
static void branch_1x1(const float *in, float *out, int C, int H, int W)
{
    /* step 1: 1x1 branch -- no spatial context, just copies each channel pixel-
     * for-pixel; captures fine-grained local features at the original resolution */
    for (int c = 0; c < C; c++) {     /* c: channel index */
        for (int h = 0; h < H; h++) { /* h: spatial row */
            for (int x = 0; x < W; x++) { /* x: spatial col */
                /* input flat index: [c][h][x] -> c*H*W + h*W + x */
                out[c*H*W + h*W + x] = in[c*H*W + h*W + x];
            }
        }
    }
}

/* dilated 3x3 (all-ones kernel, depthwise) */
static void branch_dilated(const float *in, float *out, int C, int H, int W, int rate)
{
    /* dilated branch: gathers 9 points spread rate apart; averaging normalises
     * for the varying cnt when border pixels fall outside the image */
    for (int c = 0; c < C; c++) {         /* c: channel index */
        for (int h = 0; h < H; h++) {     /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */

                float sum = 0.0f;
                int   cnt = 0;            /* cnt: valid (in-bounds) tap count */
                for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                        /* gap formula: same (kh-1)*rate pattern used by all dilated convs */
                        int ih = h + (kh - 1) * rate; /* input row with dilation gap */
                        int iw = x + (kw - 1) * rate; /* input col with dilation gap */
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            continue; /* outside image boundary = zero padding */
                        }
                        /* input flat index: [c][ih][iw] -> c*H*W + ih*W + iw */
                        sum += in[c*H*W + ih*W + iw];
                        cnt++;
                    }
                }
                out[c*H*W + h*W + x] = (cnt > 0) ? sum / cnt : 0.0f;
            }
        }
    }
}

/* global average pool -- collapses H x W to a single scalar, broadcast back */
static void branch_gap(const float *in, float *out, int C, int H, int W)
{
    /* GAP branch: reduces the entire spatial map to one scalar per channel,
     * giving a global context signal; that scalar is broadcast back to every
     * output position so downstream layers can see the whole-image average */
    for (int c = 0; c < C; c++) { /* c: channel index */

        /* step 1: sum all H*W values for this channel */
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int x = 0; x < W; x++) {
                /* input flat index: [c][h][x] -> c*H*W + h*W + x */
                sum += in[c*H*W + h*W + x];
            }
        }
        float avg = sum / (float)(H * W); /* avg: global mean for channel c */

        /* step 2: broadcast avg back to every spatial position */
        for (int h = 0; h < H; h++) {
            for (int x = 0; x < W; x++) {
                out[c*H*W + h*W + x] = avg;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 7, W = 7;

    float *in  = calloc(C*H*W, sizeof(float));
    float *b1  = calloc(C*H*W, sizeof(float));
    float *b2  = calloc(C*H*W, sizeof(float));
    float *b4  = calloc(C*H*W, sizeof(float));
    float *b6  = calloc(C*H*W, sizeof(float));
    float *bgap= calloc(C*H*W, sizeof(float));

    /* diagonal gradient: value = row + col */
    for (int h = 0; h < H; h++) {
        for (int x = 0; x < W; x++) {
            in[h*W + x] = (float)(h + x);
        }
    }

    branch_1x1   (in, b1,   C, H, W);
    branch_dilated(in, b2,  C, H, W, 2);
    branch_dilated(in, b4,  C, H, W, 4);
    branch_dilated(in, b6,  C, H, W, 6);
    branch_gap   (in, bgap, C, H, W);

    show("INPUT  (diagonal gradient)", in,   C, H, W);
    show("BRANCH 1x1  (identity)",     b1,   C, H, W);
    show("BRANCH dil2 (avg over 5x5)", b2,   C, H, W);
    show("BRANCH dil4 (avg over 9x9)", b4,   C, H, W);
    show("BRANCH dil6 (avg over 13x13)",b6,  C, H, W);
    show("BRANCH GAP  (global avg broadcast)", bgap, C, H, W);

    /* global avg of a diagonal ramp H=7,W=7: mean value = (0+12)/2 = 6.0 */
    printf("[16] check: GAP value = %.1f (expected 6.0)\n", bgap[0]);

    free(in); free(b1); free(b2); free(b4); free(b6); free(bgap);
    return 0;
}
