/* ============================================================================
 * [35] Conv -> Group Norm -> ReLU
 * ----------------------------------------------------------------------------
 * SHAPE: same spatial footprint as the conv (3x3 here). GN differs from BN
 * only in what it normalizes OVER: BN normalizes across the batch dimension;
 * GN normalizes within groups of channels WITHIN a single sample. This makes
 * it stable with small batch sizes (e.g. detection models with batch=2).
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][3][3]
 *   GN    : gamma[C_out], beta[C_out]   (num_groups divides C_out)
 *   Output: [C_out][H][W]  = ReLU( GN( Conv3x3(x) ) )
 *
 * FLOPs: same as BN at inference. GN is slightly costlier at training.
 *
 * DEMO: a gradient input. After conv and GN (2 groups), print the normalized
 * output -- each group of channels has zero mean and unit variance independently.
 *
 * Build:  gcc 35_conv_gn_relu.c -o 35_conv_gn_relu -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K   3
#define PAD 1

static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s  [C=%d H=%d W=%d]\n", title, C, H, W);
    for (int c = 0; c < C; c++) {
        if (C > 1) { printf("  channel %d:\n", c); }
        for (int h = 0; h < H; h++) {
            printf("   ");
            for (int w = 0; w < W; w++) { printf("%8.4f", m[c*H*W+h*W+w]); }
            printf("\n");
        }
    }
    printf("\n");
}

static void conv3x3(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    /* standard 3x3 conv with PAD=1 (same-padding): accumulate over all
       input channels and the 3x3 kernel neighbourhood for each output pixel */
    for (int co=0;co<Co;co++) {         /* co: output channel */
        for (int h=0;h<H;h++) {         /* h: output row */
            for (int x=0;x<W;x++) {     /* x: output column */
                float s=0.0f;           /* accumulator for this output pixel */
                for (int ci=0;ci<Ci;ci++) {   /* ci: input channel */
                    for (int kh=0;kh<K;kh++) {    /* kh: kernel row (0..K-1) */
                        for (int kw=0;kw<K;kw++) { /* kw: kernel col (0..K-1) */
                            int ih=h+kh-PAD, iw=x+kw-PAD; /* input coords with pad offset */
                            if (ih<0||ih>=H||iw<0||iw>=W) { continue; } /* outside image boundary = zero padding */
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*Ci+ci)*K+kh)*K+kw */
                            s += in[ci*H*W+ih*W+iw]*w[((co*Ci+ci)*K+kh)*K+kw];
                        }
                    }
                }
                out[co*H*W+h*W+x]=s; /* store to output flat index: [co][h][x] */
            }
        }
    }
}

/* group norm: normalize within each group of (C/G) channels over H*W.
   Key mechanic: split channels into G equal groups; for each group compute
   mean and variance across all (cpg * H * W) elements, then normalize each
   element as xhat = (x - mean) / std, apply per-channel affine: gamma*xhat + beta.
   Unlike BN (batch axis) or IN (single channel), GN normalizes over a channel
   group -- stable for small batch sizes. */
static void group_norm(float *x, const float *gamma, const float *beta,
                       int C, int G, int H, int W, float eps)
{
    int cpg = C / G;                     /* channels per group */
    for (int g = 0; g < G; g++) {        /* g: group index (0..G-1) */

        int c_start = g * cpg;           /* first channel in this group */
        int n       = cpg * H * W;       /* total elements in this group */

        /* step 1: accumulate sum for mean over all channels and spatial positions in group */
        float s = 0.0f;
        for (int c = c_start; c < c_start + cpg; c++) {
            for (int i = 0; i < H*W; i++) { s += x[c*H*W+i]; }
        }
        float mean = s / (float)n;       /* group mean */

        /* step 2: accumulate squared deviations for variance */
        float v = 0.0f;
        for (int c = c_start; c < c_start + cpg; c++) {
            for (int i = 0; i < H*W; i++) {
                float d = x[c*H*W+i] - mean;
                v += d*d;
            }
        }
        float std = sqrtf(v / (float)n + eps); /* group std; eps prevents div-by-zero */

        /* step 3: normalize each element, then apply learned affine (gamma, beta per channel) */
        /* normalize formula: xhat = (x - mean) / std ; out = gamma[c] * xhat + beta[c] */
        for (int c = c_start; c < c_start + cpg; c++) {
            for (int i = 0; i < H*W; i++) {
                float xhat = (x[c*H*W+i] - mean) / std; /* zero-mean unit-var within group */
                x[c*H*W+i] = gamma[c] * xhat + beta[c]; /* learned scale and shift per channel */
            }
        }
    }
}

static void relu(float *x, int n)
{
    for (int i=0;i<n;i++) { if (x[i]<0.0f) { x[i]=0.0f; } }
}

int main(void)
{
    int Ci=1, Co=4, G=2, H=4, W=4;

    float *in  = calloc(Ci*H*W,    sizeof(float));
    float *w   = calloc(Co*Ci*K*K, sizeof(float));
    float *out = calloc(Co*H*W,    sizeof(float));
    float gamma[4]={1.0f,1.0f,1.0f,1.0f};
    float beta [4]={0.0f,0.0f,0.0f,0.0f};

    for (int h=0;h<H;h++) {
        for (int x=0;x<W;x++) { in[h*W+x]=(float)(h+x); }  /* ramp */
    }
    for (int i=0;i<Co*Ci*K*K;i++) { w[i]=1.0f; }

    show("INPUT  (diagonal ramp)", in, Ci, H, W);
    conv3x3(in, w, out, Ci, Co, H, W);
    show("AFTER CONV3x3 (box sum, 4 output channels)", out, Co, H, W);
    group_norm(out, gamma, beta, Co, G, H, W, 1e-5f);
    show("AFTER GROUP NORM (G=2: channels 0-1 normalized together, 2-3 together)", out, Co, H, W);
    relu(out, Co*H*W);
    show("AFTER RELU", out, Co, H, W);

    printf("[35] check: GN makes each group zero-mean unit-var independent of batch size\n");

    free(in); free(w); free(out);
    return 0;
}
