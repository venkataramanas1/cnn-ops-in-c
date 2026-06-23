/*
 * Operator  : 2x2 Average Pooling, stride 2
 * Input     : [C][H][W]
 * Output    : [C][Ho][Wo]  Ho=(H-2)/2+1  Wo=(W-2)/2+1
 * FLOPs     : C * Ho * Wo * (4 adds + 1 div) = 5 * C * Ho * Wo
 * Models    : SqueezeNet, some EfficientNet stages — smooth downsampling
 * DEMO      : 1 channel, 4x4 ramp input [1..16]
 *             avg pool gives smooth downsampled version
 * Check     : top-left output = (1+2+5+6)/4 = 3.5
 * Build     : gcc 02_avgpool2x2_s2.c -O2 -o avgpool2x2_s2 -lm
 */

#include <stdio.h>

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
 * avgpool2x2_s2
 *   in  : [C][H][W] input feature map
 *   out : [C][Ho][Wo] output, Ho=(H-2)/2+1, Wo=(W-2)/2+1
 *   For each output cell (oh, ow) average the 2x2 window
 *   at input rows [oh*2 .. oh*2+1], cols [ow*2 .. ow*2+1].
 */
static void avgpool2x2_s2(const float *in, float *out,
                           int C, int H, int W)
{
    int Ho = (H - 2) / 2 + 1;
    int Wo = (W - 2) / 2 + 1;

    int c, oh, ow, kh, kw;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < Ho; oh++)       /* output row */
        {
            for (ow = 0; ow < Wo; ow++)   /* output col */
            {
                float sum = 0.0f;         /* accumulate window sum */
                for (kh = 0; kh < 2; kh++) /* kernel row offset */
                {
                    for (kw = 0; kw < 2; kw++) /* kernel col offset */
                    {
                        int ih = oh * 2 + kh;   /* stride=2 maps to input row */
                        int iw = ow * 2 + kw;   /* stride=2 maps to input col */
                        sum += in[c * H * W + ih * W + iw];
                    }
                }
                /* divide by window area to get average */
                out[c * Ho * Wo + oh * Wo + ow] = sum / 4.0f;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 4, W = 4;
    float in[1 * 4 * 4];
    int i;
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)(i + 1);  /* 1, 2, 3, ... 16 */
    }

    int Ho = (H - 2) / 2 + 1;   /* = 2 */
    int Wo = (W - 2) / 2 + 1;   /* = 2 */
    float out[1 * 2 * 2];

    show("INPUT [1][4][4] ramp 1..16:", in, C, H, W);
    avgpool2x2_s2(in, out, C, H, W);
    show("OUTPUT [1][2][2] after 2x2 avgpool stride 2:", out, C, Ho, Wo);

    /* correctness check: top-left = (1+2+5+6)/4 = 3.5 */
    float expected = 3.5f;
    float got = out[0];
    printf("CHECK: out[0][0][0] = %.2f  expected %.2f  %s\n",
           got, expected, (got == expected) ? "PASS" : "FAIL");
    return 0;
}
