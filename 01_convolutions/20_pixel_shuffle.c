/* ============================================================================
 * [20] PixelShuffle (Sub-pixel Convolution)  --  super-resolution upsample
 * ----------------------------------------------------------------------------
 * SHAPE: NO convolution kernel from the top. It is a pure REARRANGEMENT.
 * Imagine a thick stack of C*r*r channels, "unfolded" like a quilt into a
 * wider, thinner grid. Each group of r*r channels becomes one output channel
 * at r times the spatial resolution.
 *
 *   Input : [C * r * r][H][W]           (r = upscale factor)
 *   (no weight tensor -- purely a reshape / permute)
 *   Output: [C][H*r][W*r]
 *
 * FLOPs: 0 (it is a data movement only -- index arithmetic, no multiply-add)
 *
 * DEMO: 4 input channels (r=2, C=1 output) containing sub-pixel values
 * [A,B,C,D]. After shuffle a 2x2 output is built:
 *
 *   ch0 -> top-left,   ch1 -> top-right
 *   ch2 -> bot-left,   ch3 -> bot-right
 *
 * Input grids get interleaved in space into a single higher-res output map.
 *
 * Build:  gcc 20_pixel_shuffle.c -o 20_pixel_shuffle -lm
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

/*
 * pixel_shuffle:
 *   in  layout: [C_out * r * r][H_in][W_in]
 *   out layout: [C_out][H_in*r][W_in*r]
 *
 *   out[c][h*r + rh][w*r + rw]  =  in[(c*r*r) + rh*r + rw][h][w]
 */
void pixel_shuffle(const float *in, float *out,
                   int C_out, int H_in, int W_in, int r)
{
    /* pixel shuffle: pure index reinterpretation -- no multiply-add at all;
     * conceptually reshape input [C_out*r*r][H][W] then transpose the r*r
     * sub-channels into spatial offsets: [C_out][r*r][H][W]
     *   -> [C_out][H][W][r*r] -> [C_out][H*r][W*r]
     * the reindex formula encodes that transpose:
     *   in_ch  = c*r*r + rh*r + rw   (which of the r*r sub-channels to read)
     *   out_h  = h*r   + rh           (interleave sub-channel rows into output rows)
     *   out_w  = w*r   + rw           (interleave sub-channel cols into output cols) */
    int H_out = H_in * r;
    int W_out = W_in * r;

    for (int c = 0; c < C_out; c++) {   /* c: output channel group */
        for (int h = 0; h < H_in; h++) { /* h: input (coarse) row */
            for (int w = 0; w < W_in; w++) { /* w: input (coarse) col */
                for (int rh = 0; rh < r; rh++) { /* rh: sub-pixel row offset (0..r-1) */
                    for (int rw = 0; rw < r; rw++) { /* rw: sub-pixel col offset (0..r-1) */

                        /* reindex: sub-channel rh*r+rw within group c maps to spatial offset */
                        int in_ch  = c * r * r + rh * r + rw; /* input channel: [g=c][Cpg=rh*r+rw] */
                        int out_h  = h * r + rh;               /* output row: coarse row scaled + sub-row */
                        int out_w  = w * r + rw;               /* output col: coarse col scaled + sub-col */

                        /* input flat index: [in_ch][h][w] -> in_ch*H_in*W_in + h*W_in + w */
                        /* output flat index: [c][out_h][out_w] -> c*H_out*W_out + out_h*W_out + out_w */
                        out[c * H_out * W_out + out_h * W_out + out_w] =
                            in[in_ch * H_in * W_in + h * W_in + w];
                    }
                }
            }
        }
    }
}

int main(void)
{
    int r = 2, C_out = 1, H_in = 2, W_in = 2;
    int C_in  = C_out * r * r;             /* = 4 */
    int H_out = H_in * r;                  /* = 4 */
    int W_out = W_in * r;                  /* = 4 */

    float *in  = calloc(C_in  * H_in  * W_in,  sizeof(float));
    float *out = calloc(C_out * H_out * W_out,  sizeof(float));

    /* each input channel holds a distinct value so we can track where it lands */
    for (int c = 0; c < C_in; c++) {
        for (int i = 0; i < H_in * W_in; i++) {
            in[c * H_in * W_in + i] = (float)(c * 10 + i + 1);
        }
    }

    show("INPUT  (4 channels, each a 2x2 sub-image)", in, C_in, H_in, W_in);
    pixel_shuffle(in, out, C_out, H_in, W_in, r);
    show("OUTPUT (1 channel, 4x4: sub-images interleaved into space)", out, C_out, H_out, W_out);

    /* top-left pixel of output = first pixel of input channel 0 */
    printf("[20] check: out[0][0] = %.1f (expected 1.0 = in ch0 px0)\n", out[0]);
    printf("[20] check: out[0][1] = %.1f (expected 11.0 = in ch1 px0)\n", out[1]);

    free(in);
    free(out);
    return 0;
}
