/* ============================================================================
 * [17] Transposed Conv 2x2  --  learnable 2x upsample
 * ----------------------------------------------------------------------------
 * SHAPE: a 2x2 square -- four cells in a 2x2 block. No center pixel. It PAINTS
 * output values at positions between existing pixels, expanding the spatial grid.
 * "Deconvolution" is a misleading name: it is really a strided convolution on a
 * zero-inserted (dilated) input. Stride in = step in OUTPUT, kernel fills gaps.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_in ][C_out][2][2]
 *   Output: [C_out][H*2][W*2]        (stride 2 -> 2x spatial upsample)
 *
 * FLOPs ~ 2 * C_in * C_out * H * W * 4   (4 = 2x2 taps)
 *
 * DEMO: a 2x2 input with values 1,2,3,4. After transposed conv with all-ones
 * weights, each input value PAINTS a 2x2 block in the output. Overlapping
 * regions SUM -- watch the middle cells accumulate from multiple input pixels.
 *
 * Build:  gcc 17_transposed_conv2x2.c -o 17_transposed_conv2x2 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K  2
#define ST 2                    /* stride (= upsample factor) */

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

/* transposed conv: for each input pixel, SCATTER its contribution via the kernel */
void transposed_conv2x2(const float *in, const float *w, float *out,
                        int C_in, int C_out, int H_in, int W_in,
                        int H_out, int W_out)
{
    /* scatter (not gather): each input pixel (h,x) PAINTS to output positions
     * (h*ST+kh, x*ST+kw) -- the kernel fans out contributions, opposite of
     * standard conv which gathers neighbourhood values into one output */

    /* zero the output first (overlapping contributions will be summed) */
    for (int i = 0; i < C_out * H_out * W_out; i++) {
        out[i] = 0.0f;
    }

    for (int ci = 0; ci < C_in; ci++) {  /* ci: input channel */
        for (int h = 0; h < H_in; h++) { /* h: input row */
            for (int x = 0; x < W_in; x++) { /* x: input col */

                /* input flat index: [ci][h][x] -> ci*H_in*W_in + h*W_in + x */
                float val = in[ci * H_in * W_in + h * W_in + x];

                for (int co = 0; co < C_out; co++) { /* co: output channel */
                    for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                            /* scatter: this input pixel (h,x) paints to output positions (h*ST+kh, x*ST+kw) */
                            int oh = h * ST + kh; /* output row = input row scaled by stride, plus kernel offset */
                            int ow = x * ST + kw; /* output col = input col scaled by stride, plus kernel offset */
                            if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) {
                                continue;
                            }
                            /* weight layout: [C_in][C_out][K][K] */
                            float weight = w[((ci * C_out + co) * K + kh) * K + kw];
                            /* output flat index: [co][oh][ow] -> co*H_out*W_out + oh*W_out + ow */
                            out[co * H_out * W_out + oh * W_out + ow] += val * weight;
                        }
                    }
                }
            }
        }
    }
}

int main(void)
{
    int C_in = 1, C_out = 1, H_in = 2, W_in = 2;
    int H_out = H_in * ST;          /* = 4 */
    int W_out = W_in * ST;          /* = 4 */

    float *in  = calloc(C_in  * H_in  * W_in,  sizeof(float));
    float *w   = calloc(C_in  * C_out * K * K,  sizeof(float));
    float *out = calloc(C_out * H_out * W_out,  sizeof(float));

    in[0] = 1.0f; in[1] = 2.0f;
    in[2] = 3.0f; in[3] = 4.0f;

    for (int i = 0; i < C_in * C_out * K * K; i++) {
        w[i] = 1.0f;                            /* all-ones: plain scatter */
    }

    show("INPUT  (2x2, values 1-4)", in, C_in, H_in, W_in);
    show("KERNEL (2x2 all-ones)", w, 1, K, K);
    transposed_conv2x2(in, w, out, C_in, C_out, H_in, W_in, H_out, W_out);
    show("OUTPUT (4x4: each input paints a 2x2 block, overlaps sum)", out, C_out, H_out, W_out);

    /* top-left corner gets only in[0]=1; center [1][1] gets in[0]+in[1]+in[2]+in[3]=10 */
    printf("[17] check: corner[0][0] = %.1f (expected 1.0), center[1][1] = %.1f (expected 10.0)\n",
           out[0], out[1 * W_out + 1]);

    free(in);
    free(w);
    free(out);
    return 0;
}
