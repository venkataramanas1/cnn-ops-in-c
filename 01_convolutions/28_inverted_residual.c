/* ============================================================================
 * [28] Inverted Residual  --  MobileNetV2 core block
 * ----------------------------------------------------------------------------
 * SHAPE: dot (expand) -> depthwise 3x3 (spatial) -> dot (compress) + skip.
 * The opposite of the standard bottleneck [26]: it is WIDE in the middle and
 * NARROW at the ends. Cheap because the costly spatial work (depthwise 3x3)
 * runs on the EXPANDED (wide) representation, but the expansion is per-channel
 * (depthwise), not dense.
 *
 *   Input  : [C_in][H][W]
 *   Weight1: [t*C_in][C_in][1][1]           (1x1 expand by factor t)
 *   Weight2: [t*C_in][3][3]                 (depthwise 3x3 in expanded space)
 *   Weight3: [C_out ][t*C_in][1][1]         (1x1 project back down)
 *   Output : [C_out][H][W]  =  project + x  (skip when C_in == C_out)
 *
 * FLOPs ~ 2*H*W*(C_in*t*C_in + 9*t*C_in + t*C_in*C_out)
 *
 * DEMO: small map, expansion t=4. Print expand->spatial->project steps.
 *
 * Build:  gcc 28_inverted_residual.c -o 28_inverted_residual -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

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

static void relu6(float *x, int n)         /* ReLU6: clamp at 6, used in MobileNetV2 */
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) { x[i] = 0.0f; }
        if (x[i] > 6.0f) { x[i] = 6.0f; }
    }
}

static void conv1x1(const float *in, const float *w, float *out, int Ci, int Co, int H, int W)
{
    /* 1x1 conv: no spatial neighborhood -- just a per-position dot product
     * across all input channels.  Acts as a learned channel mixing matrix. */
    for (int co = 0; co < Co; co++) {        /* co: output channel */
        for (int h = 0; h < H; h++) {        /* h: spatial row */
            for (int x = 0; x < W; x++) {   /* x: spatial col */
                float s = 0.0f;
                for (int ci = 0; ci < Ci; ci++) { /* ci: input channel */
                    /* weight layout: w[co][ci] = w[co*Ci + ci] */
                    s += in[ci*H*W+h*W+x] * w[co*Ci+ci];
                }
                out[co*H*W+h*W+x] = s;
            }
        }
    }
}

static void depthwise3x3(const float *in, const float *w, float *out, int C, int H, int W)
{
    /* depthwise conv: each channel c has its OWN 3x3 kernel; no cross-channel mixing.
     * This is the cheap spatial step -- C kernels of size 3x3 instead of Co*Ci kernels. */
    for (int c = 0; c < C; c++) {            /* c: channel (kernel is per-channel) */
        for (int h = 0; h < H; h++) {        /* h: output row */
            for (int x = 0; x < W; x++) {   /* x: output col */
                float s = 0.0f;
                for (int kh = 0; kh < K; kh++) {    /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */
                        int ih = h+kh-PAD, iw = x+kw-PAD; /* map to input coords with padding offset */
                        if (ih<0||ih>=H||iw<0||iw>=W) { continue; } /* outside image boundary = zero padding */
                        /* input flat index: [c][ih][iw] -> c*H*W + ih*W + iw */
                        /* weight flat index: [c][kh][kw] -> c*K*K + kh*K + kw */
                        s += in[c*H*W+ih*W+iw] * w[c*K*K+kh*K+kw];
                    }
                }
                out[c*H*W+h*W+x] = s;
            }
        }
    }
}

int main(void)
{
    int Cin = 2, t = 4, H = 4, W = 4;
    int Cexp = Cin * t;   /* = 8 (wide middle) */
    int Cout = Cin;       /* same as input for skip identity */

    float *x   = calloc(Cin*H*W,   sizeof(float));
    float *we  = calloc(Cexp*Cin,  sizeof(float));
    float *exp = calloc(Cexp*H*W,  sizeof(float));
    float *wd  = calloc(Cexp*K*K,  sizeof(float));
    float *dw  = calloc(Cexp*H*W,  sizeof(float));
    float *wp  = calloc(Cout*Cexp, sizeof(float));
    float *proj= calloc(Cout*H*W,  sizeof(float));
    float *out = calloc(Cout*H*W,  sizeof(float));

    for (int i=0;i<Cin*H*W;i++) { x[i]=1.0f; }
    for (int i=0;i<Cexp*Cin;i++) { we[i]=1.0f; }          /* expand: sum input channels */
    for (int i=0;i<Cexp*K*K;i++) { wd[i]=1.0f/(K*K); }   /* depthwise: spatial avg */
    for (int i=0;i<Cout*Cexp;i++) { wp[i]=1.0f/Cexp; }   /* project: avg across expanded */

    conv1x1   (x,   we, exp,  Cin,  Cexp, H, W);
    relu6(exp, Cexp*H*W);
    show("INPUT  x", x, Cin, H, W);
    show("EXPANDED (x2=8ch, relu6 applied)", exp, Cexp, H, W);
    depthwise3x3(exp, wd, dw, Cexp, H, W);
    relu6(dw, Cexp*H*W);
    show("AFTER DEPTHWISE 3x3 (spatial avg in wide space)", dw, Cexp, H, W);
    conv1x1(dw, wp, proj, Cexp, Cout, H, W);
    show("PROJECTED (back to 2ch)", proj, Cout, H, W);

    for (int i=0;i<Cout*H*W;i++) { out[i] = proj[i] + x[i]; }
    show("OUTPUT = proj + x  (inverted residual skip)", out, Cout, H, W);

    printf("[28] check: out center = %.4f (expected ~3.0)\n", out[1*W+1]);

    free(x);free(we);free(exp);free(wd);free(dw);free(wp);free(proj);free(out);
    return 0;
}
