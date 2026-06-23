/*
 * Operator  : Adaptive Average Pooling
 * Input     : [C][H][W]  (any resolution)
 * Output    : [C][Ht][Wt]  (user-specified target size)
 * FLOPs     : C * Ht * Wt * (window_h * window_w adds + 1 div)
 * Models    : PyTorch AdaptiveAvgPool2d used in ResNet50 final pool —
 *             allows flexible input resolutions without fixing kernel/stride
 * NOTE      : adaptive: window size = ceil(H/Ht); no need to know stride
 *             in advance. Window for output oh = [oh*H/Ht .. (oh+1)*H/Ht)
 * DEMO      : 6x6 input → 2x2 output (windows of size 3x3)
 * Check     : out[0][0][0] = mean of top-left 3x3 block
 * Build     : gcc 05_adaptive_avg_pool.c -O2 -o adaptive_avg_pool -lm
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
 * adaptive_avg_pool
 *   in   : [C][H][W] input feature map
 *   out  : [C][Ht][Wt] output at target resolution
 *   Ht, Wt : target output height and width
 *
 *   Window for output row oh spans input rows [oh*H/Ht .. (oh+1)*H/Ht).
 *   This lets the op accept any input resolution without fixing stride.
 *   PyTorch's AdaptiveAvgPool2d uses the same formula.
 */
static void adaptive_avg_pool(const float *in, float *out,
                               int C, int H, int W,
                               int Ht, int Wt)
{
    int c, oh, ow, ih, iw;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < Ht; oh++)        /* output row */
        {
            /* window boundaries in input rows — integer division floors */
            int h_start = oh * H / Ht;           /* first input row in window */
            int h_end   = (oh + 1) * H / Ht;     /* one past last input row */

            for (ow = 0; ow < Wt; ow++)    /* output col */
            {
                int w_start = ow * W / Wt;        /* first input col in window */
                int w_end   = (ow + 1) * W / Wt;  /* one past last input col */

                float sum = 0.0f;
                int count = 0;             /* actual number of elements in window */

                for (ih = h_start; ih < h_end; ih++)
                {
                    for (iw = w_start; iw < w_end; iw++)
                    {
                        sum += in[c * H * W + ih * W + iw];
                        count++;
                    }
                }
                /* divide by actual window size, not assumed kernel size */
                out[c * Ht * Wt + oh * Wt + ow] = (count > 0) ? sum / (float)count : 0.0f;
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 6, W = 6;
    int Ht = 2, Wt = 2;   /* target output size */
    float in[1 * 6 * 6];
    int i;

    /* ramp [1..36] */
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)(i + 1);
    }

    float out[1 * 2 * 2];

    show("INPUT [1][6][6] ramp 1..36:", in, C, H, W);
    adaptive_avg_pool(in, out, C, H, W, Ht, Wt);
    show("OUTPUT [1][2][2] after adaptive avg pool (6x6 -> 2x2):", out, C, Ht, Wt);

    /*
     * Window [0][0]: input rows 0..2, cols 0..2 (3x3 block)
     * values: 1,2,3, 7,8,9, 13,14,15 — sum=72, mean=8.0
     */
    float expected = 8.0f;
    float got = out[0];
    printf("CHECK: out[0][0][0] = %.2f  expected %.2f  %s\n",
           got, expected, (got == expected) ? "PASS" : "FAIL");
    return 0;
}
