/*
 * 11_channel_shuffle_g2.c
 *
 * WHAT CHANGES:  Channel order — interleave G=2 groups of channels.
 *                [G][C/G][H][W] → transpose to [C/G][G][H][W] → flatten back to [C][H][W].
 * WHAT STAYS:    All pixel values; spatial dims; channel count.
 *
 * WHERE IN MODELS:
 *   - ShuffleNetV1: after each grouped pointwise conv, channel shuffle is
 *     applied so that the next grouped conv can see channels from all groups.
 *     Without shuffle, groups are siloed — each group only ever processes
 *     its own subset of channels, killing representational power.
 *   - ShuffleNetV2: uses a simpler split-shuffle-concat variant; the shuffle
 *     is what makes the two branches exchange information each block.
 *
 * WHY GROUPED CONV ALONE IS NOT ENOUGH:
 *   A grouped conv with G=2 splits channels into two halves and convolves
 *   each independently — like having two separate small networks.  The
 *   outputs have no cross-group information.  Channel shuffle interleaves
 *   them so the NEXT grouped conv receives a mix from both groups.
 *
 * DEMO:
 *   C=4, G=2 → group 0 has channels 0-1, group 1 has channels 2-3.
 *   Before shuffle: [ch0_vals, ch1_vals, ch2_vals, ch3_vals]
 *   After  shuffle: [ch0_vals, ch2_vals, ch1_vals, ch3_vals]  (interleaved)
 *   H=W=3 for clarity.
 *
 * Build:
 *   gcc -O2 -o 11_channel_shuffle_g2 11_channel_shuffle_g2.c && ./11_channel_shuffle_g2
 */

#include <stdio.h>

#define C  4    /* total channels */
#define G  2    /* number of groups */
#define CG (C / G)  /* channels per group = 2 */
#define H  3
#define W  3

static void show_tensor(const float *t, const char *label)
{
    int c, h, w;
    printf("%s [C=%d][H=%d][W=%d]:\n", label, C, H, W);
    for (c = 0; c < C; c++)
    {
        printf("  ch%d (grp%d): ", c, c / CG);
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                printf("%4.0f", t[c * H * W + h * W + w]);
            }
            printf(" | ");
        }
        printf("\n");
    }
}

/*
 * channel_shuffle_g2 — interleave G groups of CG channels each.
 *
 * Conceptual reshape + transpose:
 *   1. View input as [G][CG][H][W]
 *      In-group channel index: ig = c / CG   (which group)
 *                              ic = c % CG   (position within group)
 *      Input flat: ig*CG*H*W + ic*H*W + h*W + w
 *
 *   2. Transpose axes 0 and 1: [CG][G][H][W]
 *      New channel index: ic*G + ig  (CG moves to slow axis)
 *
 *   3. Flatten back to [C][H][W]
 *      Output channel: c_out = ic * G + ig
 *
 * WHY transposing [G][CG] → [CG][G] interleaves:
 *   Original order: g0c0, g0c1, g1c0, g1c1  (group-major)
 *   After transpose: g0c0, g1c0, g0c1, g1c1  (channel-major within group)
 *   Every G-th output channel belongs to the same original group — they
 *   are now spread evenly, so the next grouped conv receives mixed inputs.
 */
static void channel_shuffle_g2(const float *in, float *out)
{
    int ig, ic, h, w;
    for (ig = 0; ig < G; ig++)         /* group index 0..G-1 */
    {
        for (ic = 0; ic < CG; ic++)    /* channel-within-group 0..CG-1 */
        {
            for (h = 0; h < H; h++)
            {
                for (w = 0; w < W; w++)
                {
                    /* Input channel: ig*CG + ic  (group-major order) */
                    int c_in  = ig * CG + ic;

                    /* Output channel: ic*G + ig  (transpose: CG becomes slow axis) */
                    int c_out = ic * G  + ig;

                    out[c_out * H * W + h * W + w] = in[c_in * H * W + h * W + w];
                }
            }
        }
    }
}

int main(void)
{
    float in[C * H * W];
    float out[C * H * W];
    int c, i;

    /* Group 0 (channels 0,1): all 1s; Group 1 (channels 2,3): all 2s */
    for (c = 0; c < C; c++)
    {
        for (i = 0; i < H * W; i++)
        {
            in[c * H * W + i] = (c < CG) ? 1.0f : 2.0f;
        }
    }

    printf("=== 11_channel_shuffle_g2 ===\n\n");
    show_tensor(in, "BEFORE shuffle (groups are siloed: 1s then 2s)");

    channel_shuffle_g2(in, out);

    printf("\n");
    show_tensor(out, "AFTER  shuffle (groups interleaved: 1s and 2s alternate)");

    printf("\nCHECK: out_ch0 should be grp0-ch0 (1s), "
           "out_ch1 should be grp1-ch0 (2s)\n"
           "  out[c=0][h=0][w=0]=%.1f (expect 1.0)"
           "  out[c=1][h=0][w=0]=%.1f (expect 2.0)\n",
           out[0 * H * W],
           out[1 * H * W]);

    return 0;
}
