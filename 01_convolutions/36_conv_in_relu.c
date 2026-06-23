/* ============================================================================
 * [36] Conv -> Instance Norm -> ReLU
 * ----------------------------------------------------------------------------
 * SHAPE: same spatial footprint as the conv. Instance Norm (IN) normalizes
 * per-sample per-channel: for each (sample, channel) pair it computes the
 * mean and var across H and W only. No batch, no channel-group dependence.
 * This makes it ideal for style transfer (each image has its own stats).
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][3][3]
 *   IN    : gamma[C_out], beta[C_out]   (per-channel affine)
 *   Output: [C_out][H][W]  = ReLU( IN( Conv3x3(x) ) )
 *
 * FLOPs: same as BN at inference.
 *
 * DEMO: gradient input. After conv and IN, each channel has zero mean and
 * unit variance independently (print both stats to verify).
 *
 * Build:  gcc 36_conv_in_relu.c -o 36_conv_in_relu -lm
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
    /* standard 3x3 conv with PAD=1 (same-padding): for each output location
       sum over all input channels and the 3x3 kernel patch */
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

/* instance norm: per channel over spatial H*W.
   Key mechanic: unlike BN (across batch) or GN (across channel group), IN
   normalizes each (sample, channel) pair independently over its H*W pixels.
   This removes instance-level contrast variations -- ideal for style transfer
   where each image should be normalized independently of all others. */
static void instance_norm(float *x, const float *gamma, const float *beta,
                          int C, int H, int W, float eps)
{
    for (int c = 0; c < C; c++) {   /* c: channel -- each channel normalized independently */
        /* step 1: accumulate mean over H*W spatial positions for this channel */
        float s = 0.0f;
        for (int i = 0; i < H*W; i++) { s += x[c*H*W+i]; }
        float mean = s / (float)(H*W); /* per-channel spatial mean */

        /* step 2: accumulate variance over H*W spatial positions */
        float v = 0.0f;
        for (int i = 0; i < H*W; i++) {
            float d = x[c*H*W+i] - mean;
            v += d*d;                  /* squared deviation from mean */
        }
        float std = sqrtf(v / (float)(H*W) + eps); /* per-channel spatial std; eps prevents div-by-zero */

        /* step 3: normalize and apply per-channel affine transform */
        /* normalize formula: xhat = (x - mean) / std ; out = gamma[c] * xhat + beta[c] */
        for (int i = 0; i < H*W; i++) {
            float xhat = (x[c*H*W+i] - mean) / std; /* zero-mean unit-var within channel */
            x[c*H*W+i] = gamma[c] * xhat + beta[c]; /* learned scale and shift per channel */
        }
    }
}

static void relu(float *x, int n)
{
    for (int i=0;i<n;i++) { if (x[i]<0.0f) { x[i]=0.0f; } }
}

int main(void)
{
    int Ci=1, Co=2, H=4, W=4;

    float *in  = calloc(Ci*H*W,    sizeof(float));
    float *w   = calloc(Co*Ci*K*K, sizeof(float));
    float *out = calloc(Co*H*W,    sizeof(float));
    float gamma[2]={1.0f,2.0f};      /* channel 1 gets 2x scale */
    float beta [2]={0.0f,0.5f};

    for (int h=0;h<H;h++) {
        for (int x=0;x<W;x++) { in[h*W+x]=(float)(h*W+x); }
    }
    for (int i=0;i<Co*Ci*K*K;i++) { w[i]=1.0f; }

    show("INPUT  (sequential 0..15)", in, Ci, H, W);
    conv3x3(in, w, out, Ci, Co, H, W);
    show("AFTER CONV3x3", out, Co, H, W);

    /* print per-channel mean and std before IN */
    for (int c=0;c<Co;c++) {
        float s=0.0f, v=0.0f;
        for (int i=0;i<H*W;i++) { s+=out[c*H*W+i]; }
        float m=s/(H*W);
        for (int i=0;i<H*W;i++) { float d=out[c*H*W+i]-m; v+=d*d; }
        printf("Before IN: ch%d mean=%.2f  std=%.2f\n", c, m, sqrtf(v/(H*W)));
    }
    printf("\n");

    instance_norm(out, gamma, beta, Co, H, W, 1e-5f);
    show("AFTER INSTANCE NORM (per channel: mean=0 std=1 then affine)", out, Co, H, W);
    relu(out, Co*H*W);
    show("AFTER RELU", out, Co, H, W);

    printf("[36] IN normalizes per-channel per-sample: no batch dependency\n");

    free(in); free(w); free(out);
    return 0;
}
