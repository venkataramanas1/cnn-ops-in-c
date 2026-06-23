/* ============================================================================
 * [29] SE Block  --  Squeeze-and-Excitation
 * ----------------------------------------------------------------------------
 * SHAPE: NO spatial kernel at all. Steps:
 *   1. SQUEEZE:  global average pool collapses each channel H x W -> 1 scalar.
 *   2. EXCITE:   two FC layers (with ReLU + Sigmoid) produce a per-channel
 *                scale weight in [0,1].
 *   3. SCALE:    multiply each output channel by its learned scale.
 * From the top the spatial map DISAPPEARS entirely, becomes a vector, then
 * comes back as a recalibration of channel importance.
 *
 *   Input : [C][H][W]
 *   Weight1: [C/r][C]      (FC reduce,  r = reduction ratio)
 *   Weight2: [C][C/r]      (FC expand)
 *   Output : [C][H][W]     (input * per-channel scale)
 *
 * FLOPs: 2 * C * (C/r) * 2   (two FC layers on the squeezed vector)
 *
 * DEMO: 4 channels with values 1,2,3,4. After SE the high-value channels get
 * higher scale weights -- they excite themselves. Print the scale vector and
 * the scaled output.
 *
 * Build:  gcc 29_se_block.c -o 29_se_block -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
                printf("%7.4f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void show_vec(const char *title, const float *v, int n)
{
    printf("%s  [%d]\n   ", title, n);
    for (int i = 0; i < n; i++) {
        printf("%8.4f", v[i]);
    }
    printf("\n\n");
}

int main(void)
{
    int C = 4, H = 3, W = 3, r = 2;
    int C_red = C / r;      /* = 2 */

    float *in    = calloc(C*H*W,      sizeof(float));
    float *gap   = calloc(C,          sizeof(float));   /* squeezed descriptor */
    float *w1    = calloc(C_red*C,    sizeof(float));
    float *z1    = calloc(C_red,      sizeof(float));
    float *w2    = calloc(C*C_red,    sizeof(float));
    float *scale = calloc(C,          sizeof(float));
    float *out   = calloc(C*H*W,      sizeof(float));

    /* channels: constant value = channel index + 1 */
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H*W; i++) {
            in[c*H*W + i] = (float)(c + 1);
        }
    }

    /* step 1: SQUEEZE -- collapse each channel's H x W spatial map to a single scalar
     * via global average pooling; produces a C-dimensional channel descriptor */
    for (int c = 0; c < C; c++) {                /* c: channel being squeezed */
        float sum = 0.0f;
        for (int i = 0; i < H*W; i++) {          /* i: flat spatial index (0..H*W-1) */
            sum += in[c*H*W + i];
        }
        gap[c] = sum / (float)(H * W);           /* gap[c]: mean activation of channel c */
    }

    /* step 2a: EXCITE (reduce) -- FC layer compresses C -> C/r with ReLU
     * learns which channel combinations are important */
    for (int i = 0; i < C_red * C; i++) {
        w1[i] = 1.0f / (float)C;                /* averaging reduce */
    }
    for (int cr = 0; cr < C_red; cr++) {         /* cr: reduced channel index (0..C/r-1) */
        float s = 0.0f;
        for (int c = 0; c < C; c++) {            /* c: input channel to this FC neuron */
            /* weight layout: w1[cr][c] = w1[cr*C + c] */
            s += gap[c] * w1[cr*C + c];
        }
        z1[cr] = (s > 0.0f) ? s : 0.0f;         /* ReLU: keep only positive activations */
    }

    /* step 2b: EXCITE (expand) -- FC layer expands C/r -> C with Sigmoid
     * produces a per-channel scale weight in [0,1] */
    for (int i = 0; i < C * C_red; i++) {
        w2[i] = 1.0f;
    }
    for (int c = 0; c < C; c++) {                /* c: output channel getting a scale weight */
        float s = 0.0f;
        for (int cr = 0; cr < C_red; cr++) {     /* cr: index into bottleneck representation */
            /* weight layout: w2[c][cr] = w2[c*C_red + cr] */
            s += z1[cr] * w2[c*C_red + cr];
        }
        scale[c] = 1.0f / (1.0f + expf(-s));    /* Sigmoid: squash logit to [0,1] scale weight */
    }

    /* step 3: SCALE -- broadcast per-channel weight back to the full spatial map */
    for (int c = 0; c < C; c++) {                /* c: channel to recalibrate */
        for (int i = 0; i < H*W; i++) {          /* i: flat spatial index */
            out[c*H*W + i] = in[c*H*W + i] * scale[c]; /* rescale each pixel by channel importance */
        }
    }

    show("INPUT  (ch0=1, ch1=2, ch2=3, ch3=4)", in, C, H, W);
    show_vec("SQUEEZED (global avg per channel)", gap, C);
    show_vec("SCALE weights (sigmoid output, higher ch -> higher weight)", scale, C);
    show("OUTPUT (input * scale per channel)", out, C, H, W);

    printf("[29] check: scale[0] < scale[3] (lower channel gets lower weight)\n");
    printf("[29] scale[0]=%.4f  scale[3]=%.4f\n", scale[0], scale[3]);

    free(in);free(gap);free(w1);free(z1);free(w2);free(scale);free(out);
    return 0;
}
