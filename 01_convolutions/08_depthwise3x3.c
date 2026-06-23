/* ============================================================================
 * [08] Depthwise Conv 3x3
 * ----------------------------------------------------------------------------
 * SHAPE: looks exactly like a 3x3 grid from the top -- but there is ONE such
 * grid PER input channel, and the channels never talk. N independent 3x3 lanes
 * running in parallel. C_out == C_in. No cross-channel mixing here at all (that
 * job is left to a following 1x1; see [10] Depthwise Separable).
 *
 *   Input : [C][H][W]
 *   Weight: [C][3][3]                (one 3x3 per channel -- no C_out/C_in pair)
 *   Output: [C][H][W]                (C_out == C_in == C)
 *
 * FLOPs ~ 2 * C * H * W * 9 -- about C times cheaper than dense 3x3 (C_in*C_out*9)
 *
 * In this demo the kernel is all-ones = a 3x3 BOX BLUR. Watch the bright spike
 * in the input spread into a soft 3x3 block in the output heat-map.
 *
 * Build:  gcc 08_depthwise3x3.c -o 08_depthwise3x3 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K 3
#define P 1

/* ---- tiny terminal visualizer: prints each channel as a clean number grid -- */
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

void depthwise3x3(const float *in, const float *w, float *out, int C, int H, int W)
{
    /* each channel is its own independent lane — no cross-channel mixing here */
    for (int c = 0; c < C; c++) {                  /* c: channel lane (C_out == C_in) */
        for (int h = 0; h < H; h++) {              /* h: output row */
            for (int x = 0; x < W; x++) {          /* x: output col */

                float sum = 0.0f;
                for (int kh = 0; kh < K; kh++) {   /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                        int ih = h + kh - P;       /* input row: output row h, shifted by kh, minus padding P */
                        int iw = x + kw - P;       /* input col: same logic column-wise */
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            continue;              /* outside image boundary = zero padding */
                        }
                        /* input flat index:  [c][ih][iw] -> c*H*W + ih*W + iw    */
                        /* weight flat index: [c][kh][kw] -> c*K*K + kh*K + kw    */
                        sum += in[c*H*W + ih*W + iw] * w[c*K*K + kh*K + kw];
                    }
                }
                /* output flat index: [c][h][x] -> c*H*W + h*W + x */
                out[c*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 7, W = 7;

    float *in  = calloc(C*H*W, sizeof(float));
    float *w   = calloc(C*K*K, sizeof(float));
    float *out = calloc(C*H*W, sizeof(float));

    /* interesting input: a single bright spike at the center (an impulse) */
    in[3*W + 3] = 9.0f;
    /* all-ones kernel => 3x3 box blur */
    for (int i = 0; i < C*K*K; i++) w[i] = 1.0f;

    show("INPUT  (impulse)", in, C, H, W);
    show("KERNEL (all-ones 3x3 = box blur)", w, C, K, K);
    depthwise3x3(in, w, out, C, H, W);
    show("OUTPUT (box-blurred: the spike spread into a 3x3 block)", out, C, H, W);

    /* correctness check: the 8 neighbors of center each get one copy of 9 */
    printf("[08] check: out[2][2] = %.1f (expected 9.0), out[3][3] = %.1f (9.0)\n",
           out[2*W + 2], out[3*W + 3]);

    free(in);
    free(w);
    free(out);
    return 0;
}
