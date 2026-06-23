/* ============================================================================
 * [04] Conv 7x7  --  three rings out from center (49 cells)
 * ----------------------------------------------------------------------------
 * SHAPE FROM THE TOP: 3x3 with TWO more rings added. Big footprint used in the
 * very first "stem" layer (ResNet, YOLOv3) usually with stride 2 to grab coarse
 * structure fast while shrinking the map. Expensive but wide.
 * Same code as conv3x3, K=7, PAD=3. This version also shows STRIDE.
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][7][7]
 *   Bias  : [C_out]
 *   Output: [C_out][Ho][Wo]          Ho = (H + 2*PAD - K)/stride + 1
 *
 * FLOPs ~ 2 * C_in * C_out * Ho * Wo * 49  (49 = 7x7 taps per channel pair)
 *
 * DEMO: an 8x8 ramp (value = row+col) is averaged by a 7x7 box kernel at
 * stride 2. Watch an 8x8 map SHRINK to 4x4 -- this is how a stem downsamples.
 *
 * Build:  gcc 04_conv7x7.c -o 04_conv7x7 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K   7
#define PAD 3   /* (K-1)/2 = 3, keeps output same size when stride=1 */

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

void conv7x7(const float *in, const float *w, const float *bias, float *out,
             int C_in, int C_out, int H, int W, int s, int Ho, int Wo)
{
    for (int co = 0; co < C_out; co++) {         /* co: output channel           */
        for (int oh = 0; oh < Ho; oh++) {         /* oh: output row               */
            for (int ow = 0; ow < Wo; ow++) {     /* ow: output column            */

                float sum = bias ? bias[co] : 0.0f;

                for (int ci = 0; ci < C_in; ci++) {      /* ci: input channel    */
                    for (int kh = 0; kh < K; kh++) {     /* kh: kernel row       */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel column    */

                            /* stride s: output step oh maps to input step oh*s   */
                            /* PAD shifts the window so output[0] is centred      */
                            int ih = oh*s + kh - PAD;   /* input row              */
                            int iw = ow*s + kw - PAD;   /* input column           */

                            /* out-of-bounds input coords = zero padding */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue;
                            }

                            /* weight flat index: [co][ci][kh][kw] */
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*C_in + ci)*K + kh)*K + kw];
                        }
                    }
                }

                /* output flat index: [co][oh][ow] -- note uses Ho,Wo not H,W */
                out[co*Ho*Wo + oh*Wo + ow] = sum;
            }
        }
    }
}

int main(void)
{
    int C_in = 1, C_out = 1, H = 8, W = 8, s = 2;
    int Ho = (H + 2*PAD - K)/s + 1;            /* = 4 */
    int Wo = (W + 2*PAD - K)/s + 1;            /* = 4 */

    float *in  = calloc(C_in*H*W,       sizeof(float));
    float *w   = calloc(C_out*C_in*K*K, sizeof(float));
    float *out = calloc(C_out*Ho*Wo,    sizeof(float));

    /* input ramp: value = row + col (a smooth diagonal gradient) */
    for (int h = 0; h < H; h++) {
        for (int x = 0; x < W; x++) {
            in[h*W + x] = (float)(h + x);
        }
    }
    /* averaging box kernel (1/49 each) */
    for (int i = 0; i < K*K; i++) {
        w[i] = 1.0f / (float)(K*K);
    }

    show("INPUT  (8x8 diagonal ramp)", in, C_in, H, W);
    show("KERNEL (7x7 averaging box, each 1/49)", w, 1, K, K);
    conv7x7(in, w, NULL, out, C_in, C_out, H, W, s, Ho, Wo);
    show("OUTPUT (4x4: smoothed AND downsampled by stride 2)", out, C_out, Ho, Wo);

    printf("[04] check: output map = %dx%d (expected 4x4)\n", Ho, Wo);

    free(in);
    free(w);
    free(out);
    return 0;
}
