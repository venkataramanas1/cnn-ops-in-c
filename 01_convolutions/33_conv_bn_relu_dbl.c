/* ============================================================================
 * [33] Conv -> BN -> ReLU  (DBL block -- YOLOv3's atom)
 * ----------------------------------------------------------------------------
 * SHAPE: the convolution kernel's shape (usually 3x3 or 1x1). BN and ReLU
 * add no spatial footprint -- they operate element-wise. Shape = conv's shape.
 * This triple is the fundamental repeating unit of YOLOv3 (called DBL:
 * Darknet Conv-BN-LeakyReLU, but here we use plain ReLU for clarity).
 *
 *   Input : [C_in ][H][W]
 *   Weight: [C_out][C_in][3][3]      (the only learned spatial parameters)
 *   BN    : gamma[C_out], beta[C_out], running_mean[C_out], running_var[C_out]
 *   Output: [C_out][H][W]   = ReLU( BN( Conv3x3(x) ) )
 *
 * FLOPs: conv 2*Ci*Co*H*W*9  +  BN ~4*Co*H*W  (negligible)
 *
 * DEMO: impulse -> 3x3 conv (box sum) -> BN (normalizes to zero-mean unit-var
 * then scales with gamma/beta) -> ReLU. See the normalized values and the
 * activation clamp.
 *
 * Build:  gcc 33_conv_bn_relu_dbl.c -o 33_conv_bn_relu_dbl -lm
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
    /* standard 3x3 conv: for each output channel co and each spatial position (h,x),
     * accumulate a weighted sum over all input channels and the 3x3 neighborhood */
    for (int co=0;co<Co;co++) {            /* co: output channel */
        for (int h=0;h<H;h++) {            /* h: output row */
            for (int x=0;x<W;x++) {        /* x: output col */
                float s=0.0f;
                for (int ci=0;ci<Ci;ci++) {        /* ci: input channel */
                    for (int kh=0;kh<K;kh++) {     /* kh: kernel row (0..K-1) */
                        for (int kw=0;kw<K;kw++) { /* kw: kernel col (0..K-1) */
                            int ih=h+kh-PAD, iw=x+kw-PAD; /* map to input coords with padding offset */
                            if (ih<0||ih>=H||iw<0||iw>=W) { continue; } /* outside image boundary = zero padding */
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*Ci+ci)*K+kh)*K+kw */
                            s += in[ci*H*W+ih*W+iw]*w[((co*Ci+ci)*K+kh)*K+kw];
                        }
                    }
                }
                out[co*H*W+h*W+x]=s;
            }
        }
    }
}

/* batch norm inference: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * normalizes conv output to zero-mean unit-variance, then rescales with learned gamma/beta.
 * this stabilizes training by decoupling the scale of each channel from the conv weights. */
static void batch_norm(float *x, const float *gamma, const float *beta,
                       const float *mean, const float *var,
                       float eps, int C, int H, int W)
{
    for (int c = 0; c < C; c++) {               /* c: channel (each has its own gamma, beta, mean, var) */
        for (int i = 0; i < H*W; i++) {         /* i: flat spatial index */
            /* normalize formula: xhat = (x - mean) / sqrt(var + eps) */
            float xhat = (x[c*H*W+i] - mean[c]) / sqrtf(var[c] + eps); /* eps: small constant to avoid div by zero */
            x[c*H*W+i] = gamma[c] * xhat + beta[c]; /* affine transform: scale by gamma, shift by beta */
        }
    }
}

static void relu(float *x, int n)
{
    /* ReLU: max(0, x) -- clamps all negatives to zero, preserves positives */
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) { x[i] = 0.0f; } /* negative BN outputs become dead (zero) */
    }
}

int main(void)
{
    int Ci=1, Co=1, H=5, W=5;

    float *in  = calloc(Ci*H*W,   sizeof(float));
    float *w   = calloc(Co*Ci*K*K,sizeof(float));
    float *out = calloc(Co*H*W,   sizeof(float));

    float gamma[1]={2.0f}, beta[1]={-1.0f};     /* scale and shift */
    float mean[1], var_[1];

    in[2*W+2] = 9.0f;                            /* impulse */
    for (int i=0;i<Co*Ci*K*K;i++) { w[i]=1.0f; } /* box sum */

    show("INPUT  (impulse)", in, Ci, H, W);
    conv3x3(in, w, out, Ci, Co, H, W);
    show("AFTER CONV3x3 (box sum of impulse)", out, Co, H, W);

    /* compute per-channel mean and var from the output (batch of 1) */
    {
        float s=0.0f, s2=0.0f;
        for (int i=0;i<H*W;i++) { s+=out[i]; s2+=out[i]*out[i]; }
        mean[0] = s/(H*W);
        var_[0] = s2/(H*W) - mean[0]*mean[0];
    }
    printf("BN params: mean=%.4f  var=%.4f  gamma=%.1f  beta=%.1f\n\n",
           mean[0], var_[0], gamma[0], beta[0]);

    batch_norm(out, gamma, beta, mean, var_, 1e-5f, Co, H, W);
    show("AFTER BN (normalized then scaled by gamma, shifted by beta)", out, Co, H, W);

    relu(out, Co*H*W);
    show("AFTER RELU (negatives clamped to 0)", out, Co, H, W);

    printf("[33] check: only the conv-activated cells survive ReLU\n");

    free(in); free(w); free(out);
    return 0;
}
