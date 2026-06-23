/*
 * Operator  : Global Max Pooling (GMP)
 * Input     : [C][H][W]
 * Output    : [C][1][1]  — picks the peak activation over all spatial positions
 * FLOPs     : C * H * W comparisons
 * Models    : CBAM channel attention branch, some classification heads
 * DEMO      : 2 channels of 3x3
 *             ch0 = [1..9], ch1 = [2,4,6,8,10,12,14,16,18]
 *             GMP picks the channel-wise peak
 * Check     : ch0 = 9.0, ch1 = 18.0
 * Build     : gcc 04_global_max_pool.c -O2 -o global_max_pool -lm
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
 * global_max_pool
 *   in  : [C][H][W] input feature map
 *   out : [C][1][1] one maximum value per channel
 *
 *   Used in CBAM channel attention: the max-pooled descriptor captures
 *   the most salient activation in each feature map.
 *   Complementary to GAP — together they form CBAM's dual-descriptor branch.
 */
static void global_max_pool(const float *in, float *out,
                             int C, int H, int W)
{
    int area = H * W;   /* total spatial positions per channel */
    int c, h, w;
    for (c = 0; c < C; c++)
    {
        float best = -FLT_MAX;   /* track running max */
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                float v = in[c * area + h * W + w];
                if (v > best)
                {
                    best = v;    /* update max whenever a larger value is found */
                }
            }
        }
        /* store peak activation for this channel */
        out[c] = best;
    }
}

int main(void)
{
    int C = 2, H = 3, W = 3;
    float in[2 * 3 * 3];
    int i;

    /* channel 0: [1, 2, 3, 4, 5, 6, 7, 8, 9] */
    for (i = 0; i < 9; i++)
    {
        in[i] = (float)(i + 1);
    }
    /* channel 1: [2, 4, 6, 8, 10, 12, 14, 16, 18] */
    for (i = 0; i < 9; i++)
    {
        in[9 + i] = (float)(2 * (i + 1));
    }

    float out[2];   /* [C][1][1] stored as [C] */

    show("INPUT [2][3][3]:", in, C, H, W);
    global_max_pool(in, out, C, H, W);
    show("OUTPUT [2][1][1] after Global Max Pool:", out, C, 1, 1);

    /* ch0 max = 9.0, ch1 max = 18.0 */
    int pass = (out[0] == 9.0f) && (out[1] == 18.0f);
    printf("CHECK: ch0=%.2f (exp 9.00)  ch1=%.2f (exp 18.00)  %s\n",
           out[0], out[1], pass ? "PASS" : "FAIL");
    return 0;
}
