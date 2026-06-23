/* ============================================================================
 * [34] Conv -> BN -> LeakyReLU  (YOLOv3's actual DBL atom)
 * ----------------------------------------------------------------------------
 * SHAPE: identical to [33]. Only the activation changes: negative values pass
 * through at alpha=0.1 instead of clamping to zero. This prevents "dying ReLU"
 * where neurons get stuck at zero and stop learning entirely.
 *
 *   Input : [C_in][H][W]
 *   Weight: [C_out][C_in][3][3]
 *   BN    : gamma, beta, mean, var
 *   Output: [C_out][H][W]  = LeakyReLU( BN( Conv3x3(x) ), alpha=0.1 )
 *
 * FLOPs: same as [33]. Only the activation formula differs.
 *
 * DEMO: same impulse. After BN, some cells are negative (below-mean outputs
 * get negative z-score). Watch LeakyReLU let them through at 10% strength
 * instead of zeroing them, compared to [33].
 *
 * Build:  gcc 34_conv_bn_leakyrelu.c -o 34_conv_bn_leakyrelu -lm
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
    /* standard 3x3 conv: same structure as [33]; accumulates weighted sums
     * over all input channels ci and the 3x3 spatial neighborhood (kh, kw) */
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

/* batch norm inference (single-channel variant): y = gamma * (x - mean) / sqrt(var + eps) + beta
 * normalizes per-channel activations to zero-mean unit-variance then applies affine transform */
static void batch_norm(float *x, float gamma, float beta,
                       float mean, float var, float eps, int n)
{
    for (int i = 0; i < n; i++) {                /* i: flat element index across H*W */
        /* normalize formula: xhat = (x - mean) / sqrt(var + eps) */
        float xhat = (x[i] - mean) / sqrtf(var + eps); /* eps: small constant to avoid div by zero */
        x[i] = gamma * xhat + beta;              /* affine transform: scale by gamma, shift by beta */
    }
}

/* LeakyReLU: f(x) = x if x >= 0, else alpha * x
 * Unlike plain ReLU, negatives are NOT zeroed -- they leak through at fraction alpha.
 * This prevents "dying ReLU" where neurons with always-negative inputs stop contributing gradients. */
static void leaky_relu(float *x, int n, float alpha)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) { x[i] *= alpha; } /* negative: scale down by alpha (e.g. 0.1) instead of zeroing */
    }
}

int main(void)
{
    int Ci=1, Co=1, H=5, W=5;
    float alpha = 0.1f;

    float *in  = calloc(Ci*H*W,    sizeof(float));
    float *w   = calloc(Co*Ci*K*K, sizeof(float));
    float *out = calloc(Co*H*W,    sizeof(float));

    in[2*W+2] = 9.0f;
    for (int i=0;i<Co*Ci*K*K;i++) { w[i]=1.0f; }

    show("INPUT  (impulse)", in, Ci, H, W);
    conv3x3(in, w, out, Ci, Co, H, W);
    show("AFTER CONV3x3", out, Co, H, W);

    float s=0.0f, s2=0.0f;
    for (int i=0;i<H*W;i++) { s+=out[i]; s2+=out[i]*out[i]; }
    float mean = s/(H*W);
    float var  = s2/(H*W) - mean*mean;

    printf("BN params: mean=%.4f  var=%.4f  gamma=2.0  beta=-1.0\n\n",
           mean, var);

    batch_norm(out, 2.0f, -1.0f, mean, var, 1e-5f, H*W);
    show("AFTER BN", out, Co, H, W);

    leaky_relu(out, H*W, alpha);
    show("AFTER LEAKY RELU (negatives * 0.1, not zeroed)", out, Co, H, W);

    printf("[34] alpha=%.1f: negative cells survive at 10%% strength\n", alpha);

    free(in); free(w); free(out);
    return 0;
}
