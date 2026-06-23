/* ============================================================================
 * [27] Dense Connection  --  DenseNet
 * ----------------------------------------------------------------------------
 * SHAPE: not a single kernel. Every layer's output is CONCATENATED to all
 * future layers' inputs. Imagine a growing bundle of wires -- every 3x3 sees
 * all previous 3x3 outputs stacked together. This encourages maximum feature
 * reuse and very strong gradient flow.
 *
 *   Layer 0 input : [C0][H][W]
 *   Layer 1 input : [C0 + k][H][W]        (C0 original + k new channels)
 *   Layer 2 input : [C0 + 2k][H][W]       (+ k more from layer 1)
 *   ...
 *   Each layer adds k "growth rate" channels to the concat bundle.
 *
 *   Weight L: [k][C_so_far][3][3]
 *   Output:   [k][H][W]   (the NEW channels produced by this layer)
 *
 * FLOPs: grows quadratically with depth -- this is the trade-off vs ResNet.
 *
 * DEMO: 3 dense layers, growth k=1, starting with C0=1. Print the channel
 * bundle at each stage to see it grow: 1 -> 2 -> 3 -> 4 channels.
 *
 * Build:  gcc 27_dense_connection.c -o 27_dense_connection -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* single dense layer: reads from 'in' (C_in channels), writes k new channels
 * to 'new_feat', then copies all to 'cat' = concat(in, new_feat) */
static void dense_layer(const float *in, int C_in, float *new_feat, int k,
                        float *cat, const float *w, int H, int W)
{
    /* step 1: 3x3 conv over ALL C_in channels (the full growing bundle),
       producing k new "growth" channels -- each layer sees every prior feature */

    /* step 1: convolve the full input bundle to produce k new channels */
    for (int co = 0; co < k; co++) { /* co: new output channel index (0..k-1) */
        for (int h = 0; h < H; h++) { /* h: output row */
            for (int x = 0; x < W; x++) { /* x: output col */

                float sum = 0.0f;
                for (int ci = 0; ci < C_in; ci++) { /* ci: input channel (all channels in bundle) */
                    for (int kh = 0; kh < K; kh++) { /* kh: kernel row (0..K-1) */
                        for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */

                            int ih = h + kh - PAD; /* ih: input row with padding offset */
                            int iw = x + kw - PAD; /* iw: input col with padding offset */
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                continue; /* outside image boundary = zero padding */
                            }
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*C_in+ci)*K+kh)*K+kw */
                            sum += in[ci*H*W + ih*W + iw]
                                 * w[((co*C_in + ci)*K + kh)*K + kw];
                        }
                    }
                }
                new_feat[co*H*W + h*W + x] = (sum > 0.0f) ? sum : 0.0f; /* relu in-place */
            }
        }
    }
    /* step 2: concatenate: cat = [in | new_feat]
       copy original bundle first, then append the k new channels behind it */
    memcpy(cat,              in,       C_in * H * W * sizeof(float)); /* copy existing bundle */
    memcpy(cat + C_in*H*W,  new_feat,    k * H * W * sizeof(float)); /* append new channels */
}

int main(void)
{
    int C0 = 1, k = 1, H = 4, W = 4, layers = 3;

    /* maximum bundle size */
    int C_max = C0 + k * layers;       /* = 4 */
    float *bundle = calloc(C_max * H * W, sizeof(float));
    float *newfeat = calloc(k * H * W,    sizeof(float));
    float *cat     = calloc(C_max * H * W,sizeof(float));

    /* initial input: constant 1.0 */
    for (int i = 0; i < C0 * H * W; i++) {
        bundle[i] = 1.0f;
    }

    printf("=== DenseNet: 3 layers, growth rate k=1, starting C=%d ===\n\n", C0);
    show("BUNDLE after layer 0 (initial input)", bundle, C0, H, W);

    int C_so_far = C0;
    for (int l = 0; l < layers; l++) {

        /* weights: all-ones, size [k][C_so_far][K][K] */
        int wsize = k * C_so_far * K * K;
        float *w  = calloc(wsize, sizeof(float));
        for (int i = 0; i < wsize; i++) {
            w[i] = 1.0f / (float)(C_so_far * 9);   /* normalize so values stay ~1 */
        }

        dense_layer(bundle, C_so_far, newfeat, k, cat, w, H, W);
        memcpy(bundle, cat, (C_so_far + k) * H * W * sizeof(float));
        C_so_far += k;

        char title[64];
        snprintf(title, sizeof(title), "BUNDLE after layer %d (C=%d)", l + 1, C_so_far);
        show(title, bundle, C_so_far, H, W);
        free(w);
    }

    printf("[27] check: final bundle has C=%d channels (expected %d)\n",
           C_so_far, C0 + k * layers);

    free(bundle); free(newfeat); free(cat);
    return 0;
}
