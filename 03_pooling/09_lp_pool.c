/*
 * Operator  : Lp Pooling (generalized pooling)
 * Input     : [C][H][W]
 * Output    : [C][Ho][Wo]  Ho=(H-K)/S+1  Wo=(W-K)/S+1  (no padding, K=2, S=2)
 * FLOPs     : C * Ho * Wo * (K*K muls for pow + K*K-1 adds + 1 pow for root)
 * Models    : Some speech/audio nets; ONNX spec LpPool op;
 *             p=2 is L2 pooling (euclidean norm), p=1 is L1 (sum of abs),
 *             p→inf converges to max pooling
 * DEMO      : 2x2 window, stride 2, on 4x4 ramp input, p=2
 *             shows behavior between avg (p=1) and max (p→inf)
 * Check     : out[0][0][0] = sqrt(1^2+2^2+5^2+6^2) = sqrt(1+4+25+36) = sqrt(66)
 * Build     : gcc 09_lp_pool.c -O2 -o lp_pool -lm
 */

#include <stdio.h>
#include <math.h>

/* print a [C][H][W] tensor as numeric grids per channel */
static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s\n", title);
    int c, h, w;
    for (c = 0; c < C; c++)
    {
        printf("  channel %d:\n", c);
        for (h = 0; h < H; h++)
        {
            printf("  ");
            for (w = 0; w < W; w++)
            {
                printf("%8.3f", m[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/*
 * lp_pool
 *   in     : [C][H][W] input feature map
 *   out    : [C][Ho][Wo] output
 *   K      : kernel/window size (square)
 *   S      : stride
 *   p      : norm order (p=1: L1/sum-abs, p=2: L2/euclidean, large p → max)
 *
 *   output = (sum_i |x_i|^p) ^ (1/p) over the K*K window
 *
 *   Lp pooling unifies avg, L2, and max pooling in one formula.
 *   p=2 is common in audio feature extractors — it emphasizes large
 *   activations more than avg but less aggressively than max.
 */
static void lp_pool(const float *in, float *out,
                    int C, int H, int W,
                    int K, int S, float p)
{
    /* output spatial dims (no padding) */
    int Ho = (H - K) / S + 1;
    int Wo = (W - K) / S + 1;

    int c, oh, ow, kh, kw;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < Ho; oh++)         /* output row */
        {
            for (ow = 0; ow < Wo; ow++)     /* output col */
            {
                float sum_p = 0.0f;         /* accumulate |x|^p over window */
                for (kh = 0; kh < K; kh++) /* kernel row offset */
                {
                    for (kw = 0; kw < K; kw++) /* kernel col offset */
                    {
                        int ih = oh * S + kh;   /* input row for this kernel tap */
                        int iw = ow * S + kw;   /* input col for this kernel tap */
                        float v = in[c * H * W + ih * W + iw];
                        /* take absolute value then raise to power p */
                        float av = (v < 0.0f) ? -v : v;
                        sum_p += powf(av, p);
                    }
                }
                /* take the p-th root: (sum |x|^p)^(1/p) */
                out[c * Ho * Wo + oh * Wo + ow] = powf(sum_p, 1.0f / p);
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 4, W = 4;
    float in[1 * 4 * 4];
    int i;

    /* ramp [1..16] */
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)(i + 1);
    }

    int K = 2, S = 2;
    float p = 2.0f;   /* L2 pooling */

    int Ho = (H - K) / S + 1;   /* = 2 */
    int Wo = (W - K) / S + 1;   /* = 2 */
    float out[1 * 2 * 2];

    show("INPUT [1][4][4] ramp 1..16:", in, C, H, W);
    lp_pool(in, out, C, H, W, K, S, p);
    show("OUTPUT [1][2][2] after L2 pool (2x2, stride 2):", out, C, Ho, Wo);

    /*
     * out[0][0][0]: window inputs = 1, 2, 5, 6
     * L2 norm = sqrt(1 + 4 + 25 + 36) = sqrt(66) ~ 8.12404
     */
    float expected = sqrtf(66.0f);
    float got = out[0];
    float diff = got - expected;
    if (diff < 0.0f) diff = -diff;
    printf("CHECK: out[0][0][0] = %.5f  expected %.5f (sqrt(66))  %s\n",
           got, expected, (diff < 1e-4f) ? "PASS" : "FAIL");

    /* also show comparison with avg pool and max for the same window */
    float avg_val = (1.0f + 2.0f + 5.0f + 6.0f) / 4.0f;  /* = 3.5 */
    float max_val = 6.0f;
    printf("  (for reference: avg=%.2f  L2=%.5f  max=%.2f)\n",
           avg_val, got, max_val);
    return 0;
}
