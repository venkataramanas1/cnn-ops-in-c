/* ============================================================================
 * [19] Bilinear Upsample + Conv 1x1  --  YOLOv3 FPN neck
 * ----------------------------------------------------------------------------
 * SHAPE: NO learned spatial kernel for the upsample step -- the 2x stretching
 * is done by bilinear interpolation (math, not weights). Then a 1x1 dot adjusts
 * channel count. YOLOv3 uses this for its FPN neck because it is parameter-free
 * in the spatial direction and avoids checkerboard artifacts.
 *
 *   Input : [C_in ][H][W]
 *   (no spatial weights for upsample -- it is interpolation)
 *   Weight: [C_out][C_in][1][1]        (channel projection after upsample)
 *   Output: [C_out][H*2][W*2]
 *
 * FLOPs ~ bilinear: 0 params, 4 muls+adds per output pixel
 *         1x1:      2 * C_in * C_out * (H*2) * (W*2)
 *
 * DEMO: a 2x2 gradient is bilinearly upsampled to 4x4, then a 1x1 halves the
 * channel count. Print the intermediate (upsampled) map so you can see the
 * smooth interpolation vs. nearest-neighbor blocky repeat.
 *
 * Build:  gcc 19_bilinear_upsample_then_1x1.c -o 19_bilinear_upsample_then_1x1 -lm
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
                printf("%7.3f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

/* bilinear 2x upsample: each output pixel is a weighted average of the 4
 * nearest input pixels based on fractional distance */
static void bilinear_upsample2x(const float *in, float *out,
                                 int C, int H_in, int W_in)
{
    /* step 1: bilinear upsample -- no weights, pure interpolation;
     * maps each output pixel back to a fractional input coordinate and blends
     * the four surrounding input samples; avoids the blocky look of nearest-
     * neighbour and the checkerboard of transposed conv */
    int H_out = H_in * 2;
    int W_out = W_in * 2;

    for (int c = 0; c < C; c++) {         /* c: channel index */
        for (int oh = 0; oh < H_out; oh++) { /* oh: output row */
            for (int ow = 0; ow < W_out; ow++) { /* ow: output col */

                /* map output pixel to fractional input coordinate using align_corners=False */
                float fy = (oh + 0.5f) / 2.0f - 0.5f; /* fy: fractional input row */
                float fx = (ow + 0.5f) / 2.0f - 0.5f; /* fx: fractional input col */
                if (fy < 0.0f) fy = 0.0f;
                if (fx < 0.0f) fx = 0.0f;

                int y0 = (int)fy;          /* y0: top input row index */
                int x0 = (int)fx;          /* x0: left input col index */
                int y1 = y0 + 1;           /* y1: bottom input row index */
                int x1 = x0 + 1;           /* x1: right input col index */
                if (y1 >= H_in) y1 = H_in - 1; /* clamp to boundary */
                if (x1 >= W_in) x1 = W_in - 1;

                float dy = fy - (float)y0; /* dy: fractional distance to bottom row */
                float dx = fx - (float)x0; /* dx: fractional distance to right col */

                /* gather the four nearest input neighbours */
                float v00 = in[c*H_in*W_in + y0*W_in + x0]; /* top-left */
                float v01 = in[c*H_in*W_in + y0*W_in + x1]; /* top-right */
                float v10 = in[c*H_in*W_in + y1*W_in + x0]; /* bottom-left */
                float v11 = in[c*H_in*W_in + y1*W_in + x1]; /* bottom-right */

                /* bilinear blend: weight each corner by (1-d) or d in each axis */
                out[c*H_out*W_out + oh*W_out + ow] =
                    v00*(1-dy)*(1-dx) + v01*(1-dy)*dx +
                    v10*dy*(1-dx)     + v11*dy*dx;
            }
        }
    }
}

static void conv1x1(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    /* step 2: 1x1 conv -- channel projection after upsample; no spatial mixing,
     * just a weighted sum across input channels at each pixel independently */
    for (int co = 0; co < Co; co++) { /* co: output channel */
        for (int h = 0; h < H; h++) { /* h: spatial row */
            for (int x = 0; x < W; x++) { /* x: spatial col */

                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) { /* ci: input channel */
                    /* input flat index: [ci][h][x] -> ci*H*W + h*W + x */
                    sum += in[ci*H*W + h*W + x] * w[co*Ci + ci];
                }
                out[co*H*W + h*W + x] = sum;
            }
        }
    }
}

int main(void)
{
    int C_in = 2, C_out = 1, H = 2, W = 2;
    int H2 = H * 2, W2 = W * 2;

    float *in  = calloc(C_in  * H  * W,  sizeof(float));
    float *mid = calloc(C_in  * H2 * W2, sizeof(float));
    float *w   = calloc(C_out * C_in,    sizeof(float));
    float *out = calloc(C_out * H2 * W2, sizeof(float));

    /* channel 0: corners 0,1,2,3;  channel 1: corners 4,5,6,7 */
    in[0*H*W + 0] = 0.0f; in[0*H*W + 1] = 1.0f;
    in[0*H*W + 2] = 2.0f; in[0*H*W + 3] = 3.0f;
    in[1*H*W + 0] = 4.0f; in[1*H*W + 1] = 5.0f;
    in[1*H*W + 2] = 6.0f; in[1*H*W + 3] = 7.0f;

    w[0] = 0.5f; w[1] = 0.5f;                   /* 1x1 averages the two channels */

    show("INPUT  (2 channels, 2x2)", in, C_in, H, W);
    bilinear_upsample2x(in, mid, C_in, H, W);
    show("AFTER BILINEAR UPSAMPLE (smooth 4x4, no weights used)", mid, C_in, H2, W2);
    show("KERNEL 1x1 (average 2 channels)", w, 1, C_out, C_in);
    conv1x1(mid, w, out, C_in, C_out, H2, W2);
    show("OUTPUT (1 channel, smoothly upsampled)", out, C_out, H2, W2);

    printf("[19] check: output map = %dx%d (expected 4x4)\n", H2, W2);

    free(in); free(mid); free(w); free(out);
    return 0;
}
