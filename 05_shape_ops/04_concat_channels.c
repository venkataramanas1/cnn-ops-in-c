/*
 * 04_concat_channels.c
 *
 * WHAT CHANGES:  Channel count grows — [C1][H][W] + [C2][H][W] → [C1+C2][H][W].
 *                Values from both tensors are preserved; spatial dims stay fixed.
 * WHAT STAYS:    H, W, and all pixel values.
 *
 * WHERE IN MODELS:
 *   - DenseNet: every dense block concatenates all previous feature maps
 *     along the channel axis — skip connections are channel-concatenated.
 *   - FPN (Feature Pyramid Network): fused multi-scale features are
 *     channel-concatenated before the prediction head.
 *   - VLA multi-modal fusion: vision token features and language token
 *     features are channel-concatenated before cross-attention, so each
 *     spatial position attends to both modalities simultaneously.
 *
 * DEMO:
 *   Tensor A [C1=2][H=3][W=3] filled with 1.0
 *   Tensor B [C2=2][H=3][W=3] filled with 2.0
 *   Output   [C=4][H=3][W=3] — first 2 channels are 1s, next 2 are 2s.
 *
 * Build:
 *   gcc -O2 -o 04_concat_channels 04_concat_channels.c && ./04_concat_channels
 */

#include <stdio.h>

#define C1 2
#define C2 2
#define C_OUT (C1 + C2)
#define H 3
#define W 3

static void show_tensor(const float *t, int C, const char *label)
{
    int c, h, w;
    printf("%s [C=%d][H=%d][W=%d]:\n", label, C, H, W);
    for (c = 0; c < C; c++)
    {
        printf("  channel %d:\n", c);
        for (h = 0; h < H; h++)
        {
            printf("    ");
            for (w = 0; w < W; w++)
            {
                /* flat index: c*H*W + h*W + w */
                printf("%5.1f", t[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/*
 * concat_channels — stack A's channels then B's channels.
 *
 * The output has C_OUT = C1+C2 channels.
 * For output channel c:
 *   if c < C1  → value comes from A at channel c
 *   if c >= C1 → value comes from B at channel (c - C1)
 *
 * WHY: channel concat is a simple block-copy in memory — A's flat block
 * (size C1*H*W) is written first, then B's flat block (size C2*H*W).
 * The channel index in the output directly encodes which source it came from.
 */
static void concat_channels(const float *a, const float *b, float *out)
{
    int c, h, w;

    /* Copy A's channels into the first C1 output channel slots */
    for (c = 0; c < C1; c++)
    {
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                /* output channel c, source channel c from A */
                out[c * H * W + h * W + w] = a[c * H * W + h * W + w];
            }
        }
    }

    /* Copy B's channels into output channel slots [C1 .. C1+C2-1] */
    for (c = 0; c < C2; c++)
    {
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                /* output channel (C1+c) ← source channel c from B */
                out[(C1 + c) * H * W + h * W + w] = b[c * H * W + h * W + w];
            }
        }
    }
}

int main(void)
{
    float a[C1 * H * W];
    float b[C2 * H * W];
    float out[C_OUT * H * W];
    int i;

    /* Tensor A: all 1s */
    for (i = 0; i < C1 * H * W; i++)
    {
        a[i] = 1.0f;
    }

    /* Tensor B: all 2s */
    for (i = 0; i < C2 * H * W; i++)
    {
        b[i] = 2.0f;
    }

    printf("=== 04_concat_channels ===\n\n");
    show_tensor(a, C1, "INPUT A");
    printf("\n");
    show_tensor(b, C2, "INPUT B");

    concat_channels(a, b, out);

    printf("\n");
    show_tensor(out, C_OUT, "OUTPUT (A concat B along channel axis)");

    printf("\nCHECK: out[c=0][h=1][w=1]=%.1f (from A, expect 1.0)"
           "  out[c=3][h=1][w=1]=%.1f (from B, expect 2.0)\n",
           out[0 * H * W + 1 * W + 1],
           out[3 * H * W + 1 * W + 1]);

    return 0;
}
