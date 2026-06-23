/*
 * Operator  : 3x3 Max Pooling, stride 2, padding 1
 * Input     : [C][H][W]
 * Output    : [C][Ho][Wo]  Ho=ceil(H/2)  Wo=ceil(W/2)  (with pad=1, stride=2)
 *             Formula: Ho = (H + 2*pad - K) / S + 1 = (H + 2 - 3) / 2 + 1 = (H-1)/2 + 1
 * FLOPs     : C * Ho * Wo * 9 comparisons
 * Models    : ResNet first stage (after 7x7 conv), MobileNet early layers
 * NOTE      : padding 1 + stride 2 is the standard ResNet spatial reduction —
 *             keeps border information that zero-padded edges preserve
 * DEMO      : 1 channel, 5x5 gradient input — show padded border behavior
 * Check     : out[0][0][0] uses top-left corner with pad zeros — verified
 * Build     : gcc 06_maxpool3x3_s2_pad1.c -O2 -o maxpool3x3_s2_pad1 -lm
 */

#include <stdio.h>
#include <float.h>

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
                printf("%7.2f", m[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/*
 * maxpool3x3_s2_pad1
 *   in   : [C][H][W] input feature map
 *   out  : [C][Ho][Wo] output
 *   pad  : 1 zero-padded border around the input
 *   stride: 2
 *
 *   padding 1 + stride 2 is the standard ResNet spatial reduction —
 *   keeps border information. Padded positions are treated as -inf
 *   (they never win the max unless all real values are negative).
 */
static void maxpool3x3_s2_pad1(const float *in, float *out,
                                int C, int H, int W)
{
    int pad = 1;
    int stride = 2;
    int K = 3;
    /* output spatial dims with padding */
    int Ho = (H + 2 * pad - K) / stride + 1;
    int Wo = (W + 2 * pad - K) / stride + 1;

    int c, oh, ow, kh, kw;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < Ho; oh++)        /* output row */
        {
            for (ow = 0; ow < Wo; ow++)    /* output col */
            {
                float best = -FLT_MAX;     /* running max, padded positions = -inf */
                for (kh = 0; kh < K; kh++) /* kernel row offset */
                {
                    for (kw = 0; kw < K; kw++) /* kernel col offset */
                    {
                        /* map to padded input coords */
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;

                        /* check if this position falls inside the real input */
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        {
                            float v = in[c * H * W + ih * W + iw];
                            if (v > best)
                            {
                                best = v;
                            }
                        }
                        /* out-of-bounds positions are padding zeros treated as -inf:
                           they contribute nothing since best starts at -FLT_MAX */
                    }
                }
                out[c * Ho * Wo + oh * Wo + ow] = best;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 5, W = 5;
    float in[1 * 5 * 5];
    int i;

    /* gradient input: value at position i = i+1 */
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)(i + 1);
    }

    int pad = 1, stride = 2, K = 3;
    int Ho = (H + 2 * pad - K) / stride + 1;   /* = (5+2-3)/2+1 = 3 */
    int Wo = (W + 2 * pad - K) / stride + 1;   /* = 3 */
    float out[1 * 3 * 3];

    show("INPUT [1][5][5] gradient 1..25:", in, C, H, W);
    maxpool3x3_s2_pad1(in, out, C, H, W);
    show("OUTPUT [1][3][3] after 3x3 maxpool stride 2 pad 1:", out, C, Ho, Wo);

    /*
     * out[0][0][0]: window centered at input (-1,-1) with pad
     * kernel covers input rows [-1..1], cols [-1..1]
     * valid cells: (0,0)=1, (0,1)=2, (1,0)=6, (1,1)=7
     * max = 7
     */
    float expected = 7.0f;
    float got = out[0];
    printf("CHECK: out[0][0][0] = %.2f  expected %.2f  %s\n",
           got, expected, (got == expected) ? "PASS" : "FAIL");
    return 0;
}
