/* ============================================================================
 * [30] CBAM Channel Attention
 * ----------------------------------------------------------------------------
 * SHAPE: same idea as SE ([29]) but uses BOTH avg-pool AND max-pool to squeeze.
 * The two descriptors are each passed through the SAME shared MLP, then summed
 * and sigmoid-gated. Richer description than SE because max captures the
 * sharpest activations while avg captures the overall response.
 *
 *   Input : [C][H][W]
 *   (squeeze) avg-pool -> [C]  AND  max-pool -> [C]
 *   (shared MLP) W1:[C/r][C], W2:[C][C/r]
 *   scale = sigmoid( MLP(avg) + MLP(max) )   shape [C]
 *   Output : [C][H][W]  = input * scale
 *
 * FLOPs: 4 * C * (C/r) * 2   (two descriptor paths through shared MLP)
 *
 * DEMO: same 4-channel constant input as [29]. Show avg-pool descriptor,
 * max-pool descriptor, their combined scale, and the scaled output.
 *
 * Build:  gcc 30_cbam_channel.c -o 30_cbam_channel -lm
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

/* shared MLP: [C] -> ReLU([C/r]) -> [C] (returns logits before sigmoid)
 * CBAM's key mechanic: the SAME weights are reused for both avg and max descriptors,
 * forcing the network to learn a single channel-importance function that works for both. */
static void mlp(const float *desc, const float *w1, const float *w2,
                float *logit, int C, int C_red)
{
    float *h = calloc(C_red, sizeof(float));  /* h: bottleneck hidden activations [C/r] */

    /* FC reduce: desc[C] -> h[C/r] with ReLU */
    for (int cr = 0; cr < C_red; cr++) {          /* cr: bottleneck neuron index (0..C/r-1) */
        float s = 0.0f;
        for (int c = 0; c < C; c++) {             /* c: input channel */
            /* weight layout: w1[cr][c] = w1[cr*C + c] */
            s += desc[c] * w1[cr*C + c];
        }
        h[cr] = (s > 0.0f) ? s : 0.0f;           /* ReLU: drop negative combinations */
    }
    /* FC expand: h[C/r] -> logit[C] (no activation yet; caller applies sigmoid) */
    for (int c = 0; c < C; c++) {                 /* c: output channel logit */
        float s = 0.0f;
        for (int cr = 0; cr < C_red; cr++) {      /* cr: bottleneck index */
            /* weight layout: w2[c][cr] = w2[c*C_red + cr] */
            s += h[cr] * w2[c*C_red + cr];
        }
        logit[c] += s;  /* accumulate: called twice (avg path + max path), then sigmoid once */
    }
    free(h);
}

int main(void)
{
    int C = 4, H = 3, W = 3, r = 2;
    int C_red = C / r;

    float *in    = calloc(C*H*W,   sizeof(float));
    float *avg   = calloc(C,       sizeof(float));
    float *mx    = calloc(C,       sizeof(float));
    float *w1    = calloc(C_red*C, sizeof(float));
    float *w2    = calloc(C*C_red, sizeof(float));
    float *logit = calloc(C,       sizeof(float));
    float *scale = calloc(C,       sizeof(float));
    float *out   = calloc(C*H*W,   sizeof(float));

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H*W; i++) {
            in[c*H*W + i] = (float)(c + 1);
        }
    }

    /* step 1: SQUEEZE -- compute avg-pool AND max-pool descriptors in one pass
     * avg captures the overall channel response; max captures the sharpest activation */
    for (int c = 0; c < C; c++) {                /* c: channel being squeezed */
        float s = 0.0f;
        float m = in[c*H*W];                     /* m: running max across spatial positions */
        for (int i = 0; i < H*W; i++) {          /* i: flat spatial index (0..H*W-1) */
            s += in[c*H*W + i];
            if (in[c*H*W + i] > m) { m = in[c*H*W + i]; } /* track per-channel spatial maximum */
        }
        avg[c] = s / (float)(H*W);               /* avg[c]: mean activation of channel c */
        mx[c]  = m;                              /* mx[c]: max activation of channel c */
    }

    /* shared MLP weights (same weights used for both avg and max paths) */
    for (int i = 0; i < C_red*C; i++) { w1[i] = 1.0f / (float)C; }
    for (int i = 0; i < C*C_red; i++) { w2[i] = 1.0f; }

    /* step 2: EXCITE -- run both descriptors through shared MLP; results accumulate
     * scale = sigmoid( MLP(avg) + MLP(max) ) */
    mlp(avg, w1, w2, logit, C, C_red);  /* avg path: contributes to logit */
    mlp(mx,  w1, w2, logit, C, C_red);  /* max path: adds to same logit buffer */

    /* sigmoid gate: convert combined logit to per-channel weight in [0,1] */
    for (int c = 0; c < C; c++) {
        scale[c] = 1.0f / (1.0f + expf(-logit[c]));
    }
    /* step 3: SCALE -- apply per-channel weight to the full spatial map */
    for (int c = 0; c < C; c++) {                /* c: channel to recalibrate */
        for (int i = 0; i < H*W; i++) {          /* i: flat spatial index */
            out[c*H*W + i] = in[c*H*W + i] * scale[c]; /* rescale each pixel by channel importance */
        }
    }

    show("INPUT  (ch0=1, ch1=2, ch2=3, ch3=4)", in, C, H, W);
    show_vec("AVG-pool descriptor", avg, C);
    show_vec("MAX-pool descriptor", mx, C);
    show_vec("SCALE (sigmoid of combined MLP)", scale, C);
    show("OUTPUT (input * channel scale)", out, C, H, W);

    printf("[30] check: scale[0] < scale[3]: %.4f < %.4f\n", scale[0], scale[3]);

    free(in);free(avg);free(mx);free(w1);free(w2);free(logit);free(scale);free(out);
    return 0;
}
