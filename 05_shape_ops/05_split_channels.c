/*
 * 05_split_channels.c
 *
 * WHAT CHANGES:  One tensor [C][H][W] splits into two halves:
 *                  A [C/2][H][W]  and  B [C/2][H][W].
 *                Inverse of concat_channels.
 * WHAT STAYS:    Every pixel value; spatial dims unchanged.
 *
 * WHERE IN MODELS:
 *   - Multi-Head Self-Attention (MHSA): the QKV projection output [3C] is
 *     split into Q, K, V each of shape [C], then further split into H heads.
 *     Each head operates on its own channel slice independently.
 *   - YOLO detection head: splits the final feature map into classification
 *     channels and regression (bbox) channels.
 *   - ShuffleNetV2: every block splits channels in two; one half bypasses
 *     the conv block (skip path), the other passes through depthwise conv.
 *
 * DEMO:
 *   Input [C=4][H=3][W=3]: channels 0-1 filled with 1.0, channels 2-3 with 2.0.
 *   After split: half-A gets the 1s, half-B gets the 2s.
 *
 * Build:
 *   gcc -O2 -o 05_split_channels 05_split_channels.c && ./05_split_channels
 */

#include <stdio.h>

#define C    4
#define HALF (C / 2)
#define H    3
#define W    3

static void show_tensor(const float *t, int nc, const char *label)
{
    int c, h, w;
    printf("%s [C=%d][H=%d][W=%d]:\n", label, nc, H, W);
    for (c = 0; c < nc; c++)
    {
        printf("  channel %d:\n", c);
        for (h = 0; h < H; h++)
        {
            printf("    ");
            for (w = 0; w < W; w++)
            {
                printf("%5.1f", t[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/*
 * split_channels — extract first HALF and second HALF channel blocks.
 *
 * For the first half (channels 0 .. HALF-1):
 *   out_a[c][h][w] = in[c][h][w]
 *   flat: out_a[c*H*W + h*W + w] = in[c*H*W + h*W + w]
 *   Channel index c is the same in source and destination.
 *
 * For the second half (channels HALF .. C-1):
 *   out_b[c'][h][w] = in[HALF + c'][h][w]
 *   flat: out_b[c'*H*W + h*W + w] = in[(HALF+c')*H*W + h*W + w]
 *   c' = c - HALF re-indexes the slice to start at 0 in out_b.
 *
 * WHY: a split is just two windowed views into the original memory block.
 * In a real runtime no copy happens; the output tensors point into
 * different offsets of the same buffer.  In C we copy to make it explicit.
 */
static void split_channels(const float *in, float *out_a, float *out_b)
{
    int c, h, w;

    /* First half: channels 0..HALF-1 */
    for (c = 0; c < HALF; c++)
    {
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                out_a[c * H * W + h * W + w] = in[c * H * W + h * W + w];
            }
        }
    }

    /* Second half: channels HALF..C-1 → re-indexed as 0..HALF-1 in out_b */
    for (c = 0; c < HALF; c++)
    {
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                /* (HALF + c) is the source channel; c is the dest channel */
                out_b[c * H * W + h * W + w] = in[(HALF + c) * H * W + h * W + w];
            }
        }
    }
}

int main(void)
{
    float in[C * H * W];
    float a[HALF * H * W];
    float b[HALF * H * W];
    int c, i;

    /* Channels 0,1 → 1.0;  channels 2,3 → 2.0 */
    for (c = 0; c < C; c++)
    {
        for (i = 0; i < H * W; i++)
        {
            in[c * H * W + i] = (c < HALF) ? 1.0f : 2.0f;
        }
    }

    printf("=== 05_split_channels ===\n\n");
    show_tensor(in, C, "INPUT");

    split_channels(in, a, b);

    printf("\n");
    show_tensor(a, HALF, "OUTPUT half-A (first 2 channels)");
    printf("\n");
    show_tensor(b, HALF, "OUTPUT half-B (last 2 channels)");

    printf("\nCHECK: A[c=0][h=1][w=1]=%.1f (expect 1.0)"
           "  B[c=0][h=1][w=1]=%.1f (expect 2.0)\n",
           a[0 * H * W + 1 * W + 1],
           b[0 * H * W + 1 * W + 1]);

    return 0;
}
