/* ============================================================================
 * [18] Transposed Conv 4x4  --  common in GAN generators
 * ----------------------------------------------------------------------------
 * SHAPE: a 4x4 square -- 16 cells. A slightly larger painting brush than 2x2.
 * Used in DCGAN generators to upsample noise into an image. At stride 2 each
 * input pixel paints a 4x4 region, with 2-cell overlap between neighbors --
 * that overlap produces "checkerboard" artifacts if weights are not learned well.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_in ][C_out][4][4]
 *   Output: [C_out][H*2][W*2]        (stride 2 -> 2x spatial upsample)
 *
 * FLOPs ~ 2 * C_in * C_out * H * W * 16
 *
 * DEMO: a 2x2 input painted with a 4x4 all-ones kernel at stride 2. Each pixel
 * covers a 4x4 patch, neighboring patches overlap by 2 cells and sum up.
 *
 * Build:  gcc 18_transposed_conv4x4.c -o 18_transposed_conv4x4 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K  4
#define ST 2

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

void transposed_conv4x4(const float *in, const float *w, float *out,
                        int C_in, int C_out, int H_in, int W_in,
                        int H_out, int W_out)
{
    /* scatter (not gather): each input pixel PAINTS a 4x4 region in the output;
     * at stride 2 neighbouring input pixels produce overlapping 4x4 patches that
     * are 2 cells apart, so the overlap zone (2 cells wide) accumulates from
     * multiple input pixels -- the source of DCGAN checkerboard artifacts */

    /* step 1: zero output so scattered contributions can accumulate cleanly */
    for (int i = 0; i < C_out * H_out * W_out; i++) {
        out[i] = 0.0f;
    }

    /* step 2: scatter each input pixel through the 4x4 kernel */
    for (int ci = 0; ci < C_in; ci++) {  /* ci: input channel */
        for (int h = 0; h < H_in; h++) { /* h: input row */
            for (int x = 0; x < W_in; x++) { /* x: input col */

                /* input flat index: [ci][h][x] -> ci*H_in*W_in + h*W_in + x */
                float val = in[ci * H_in * W_in + h * W_in + x];

                for (int co = 0; co < C_out; co++) { /* co: output channel */
                    for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                            /* scatter: this input pixel (h,x) paints to output positions (h*ST+kh-1, x*ST+kw-1) */
                            int oh = h * ST + kh - 1;   /* -1 pad keeps output size */
                            int ow = x * ST + kw - 1;
                            if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) {
                                continue; /* outside image boundary = zero padding */
                            }
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
    int H_out = H_in * ST;   /* = 4 */
    int W_out = W_in * ST;   /* = 4 */

    float *in  = calloc(C_in  * H_in  * W_in,  sizeof(float));
    float *w   = calloc(C_in  * C_out * K * K,  sizeof(float));
    float *out = calloc(C_out * H_out * W_out,  sizeof(float));

    in[0] = 1.0f; in[1] = 1.0f;
    in[2] = 1.0f; in[3] = 1.0f;

    for (int i = 0; i < C_in * C_out * K * K; i++) {
        w[i] = 1.0f;
    }

    show("INPUT  (2x2, all 1s)", in, C_in, H_in, W_in);
    show("KERNEL (4x4 all-ones)", w, 1, K, K);
    transposed_conv4x4(in, w, out, C_in, C_out, H_in, W_in, H_out, W_out);
    show("OUTPUT (4x4: overlapping 4x4 patches, center summed from 4 inputs)", out, C_out, H_out, W_out);

    printf("[18] check: output map = %dx%d (expected 4x4)\n", H_out, W_out);

    free(in);
    free(w);
    free(out);
    return 0;
}
