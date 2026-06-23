/*
 * Operator  : 3x3 Average Pooling, stride 1 (no padding)
 * Input     : [C][H][W]
 * Output    : [C][Ho][Wo]  Ho=H-2  Wo=W-2  (valid convolution footprint)
 * FLOPs     : C * Ho * Wo * (9 adds + 1 div)
 * Models    : Some normalization-free nets use avg pool for local smoothing;
 *             blur pooling (anti-aliasing) layers in modern classifiers
 * DEMO      : 1 channel, 5x5 checkerboard [1,0 alternating] —
 *             averaging kills the high-frequency pattern
 * Check     : center output = avg of 3x3 center block = 5/9 ~ 0.556
 * Build     : gcc 07_avg_pool_3x3_s1.c -O2 -o avg_pool_3x3_s1 -lm
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
                printf("%7.3f", m[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/*
 * avg_pool_3x3_s1
 *   in  : [C][H][W] input feature map
 *   out : [C][Ho][Wo] output, Ho=H-2, Wo=W-2 (no padding, stride 1)
 *
 *   Stride=1 means no spatial reduction — this is local smoothing.
 *   Each output is the mean of a 3x3 neighborhood.
 *   High-frequency checkerboard patterns average to ~0.5 (smooth output).
 */
static void avg_pool_3x3_s1(const float *in, float *out,
                              int C, int H, int W)
{
    /* output dims: no padding, kernel 3, stride 1 */
    int Ho = H - 2;
    int Wo = W - 2;
    int K = 3;

    int c, oh, ow, kh, kw;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < Ho; oh++)        /* output row */
        {
            for (ow = 0; ow < Wo; ow++)    /* output col */
            {
                float sum = 0.0f;          /* accumulate 3x3 window */
                for (kh = 0; kh < K; kh++) /* kernel row offset */
                {
                    for (kw = 0; kw < K; kw++) /* kernel col offset */
                    {
                        int ih = oh + kh;   /* stride=1: output pos = input pos directly */
                        int iw = ow + kw;
                        sum += in[c * H * W + ih * W + iw];
                    }
                }
                /* average over 9 elements — smooths high-frequency content */
                out[c * Ho * Wo + oh * Wo + ow] = sum / 9.0f;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 5, W = 5;
    float in[1 * 5 * 5];
    int h, w;

    /* checkerboard: 1 where (h+w) is even, 0 where (h+w) is odd */
    for (h = 0; h < H; h++)
    {
        for (w = 0; w < W; w++)
        {
            in[h * W + w] = (float)((h + w) % 2 == 0 ? 1 : 0);
        }
    }

    int Ho = H - 2;   /* = 3 */
    int Wo = W - 2;   /* = 3 */
    float out[1 * 3 * 3];

    show("INPUT [1][5][5] checkerboard (1=white, 0=black):", in, C, H, W);
    avg_pool_3x3_s1(in, out, C, H, W);
    show("OUTPUT [1][3][3] after 3x3 avgpool stride 1 (smoothed):", out, C, Ho, Wo);

    /*
     * Center output out[0][1][1]: window rows 1..3, cols 1..3
     * (h+w)%2==0 -> 1 else 0:
     *   (1,1)=1,(1,2)=0,(1,3)=1, (2,1)=0,(2,2)=1,(2,3)=0, (3,1)=1,(3,2)=0,(3,3)=1
     *   sum=5, mean=5/9
     */
    float expected = 5.0f / 9.0f;
    float got = out[0 * Ho * Wo + 1 * Wo + 1];   /* center cell */
    float diff = got - expected;
    if (diff < 0.0f) diff = -diff;
    printf("CHECK: out[0][1][1] = %.4f  expected %.4f  %s\n",
           got, expected, (diff < 1e-5f) ? "PASS" : "FAIL");
    return 0;
}
