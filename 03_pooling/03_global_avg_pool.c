/*
 * Operator  : Global Average Pooling (GAP)
 * Input     : [C][H][W]
 * Output    : [C][1][1]  — collapses all spatial dims to a single value per channel
 * FLOPs     : C * H * W additions + C divisions
 * Models    : MobileNet/EfficientNet classifier head, ResNet final pool before FC,
 *             ALL VLA vision encoders before action head
 * NOTE      : GAP makes the network spatially invariant — used in MobileNet,
 *             EfficientNet, and every VLA vision encoder
 * DEMO      : 2 channels of 3x3
 *             ch0 = [1..9], ch1 = [2,4,6,8,10,12,14,16,18]
 *             GAP = mean of each channel
 * Check     : ch0 = 5.0, ch1 = 10.0
 * Build     : gcc 03_global_avg_pool.c -O2 -o global_avg_pool -lm
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
 * global_avg_pool
 *   in  : [C][H][W] input feature map
 *   out : [C][1][1] one mean value per channel
 *
 *   GAP makes the network spatially invariant — used in MobileNet,
 *   EfficientNet, and every VLA vision encoder before the action head.
 *   Replaces flatten+FC with a single scalar per channel.
 */
static void global_avg_pool(const float *in, float *out,
                             int C, int H, int W)
{
    int area = H * W;   /* total spatial positions per channel */
    int c, h, w;
    for (c = 0; c < C; c++)
    {
        float sum = 0.0f;   /* accumulate all spatial positions */
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                sum += in[c * area + h * W + w];
            }
        }
        /* divide by area to get channel mean — output is [C][1][1] */
        out[c] = sum / (float)area;
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

    float out[2];  /* [C][1][1] stored as [C] */

    show("INPUT [2][3][3]:", in, C, H, W);
    global_avg_pool(in, out, C, H, W);
    show("OUTPUT [2][1][1] after Global Avg Pool:", out, C, 1, 1);

    /* ch0 mean = (1+2+...+9)/9 = 45/9 = 5.0 */
    /* ch1 mean = (2+4+...+18)/9 = 90/9 = 10.0 */
    int pass = (out[0] == 5.0f) && (out[1] == 10.0f);
    printf("CHECK: ch0=%.2f (exp 5.00)  ch1=%.2f (exp 10.00)  %s\n",
           out[0], out[1], pass ? "PASS" : "FAIL");
    return 0;
}
