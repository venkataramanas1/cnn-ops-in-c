/* ============================================================================
 * [24] Channel Shuffle  --  ShuffleNet's cross-group communication trick
 * ----------------------------------------------------------------------------
 * SHAPE: NO kernel shape at all. It is a REINDEX operation -- like physically
 * shuffling a deck of cards. No spatial footprint, no multiply-add.
 * Purpose: after grouped conv ([21],[22]), channels within each group have
 * mixed but groups have NOT exchanged information. Channel shuffle interleaves
 * the groups so the NEXT grouped conv sees cross-group inputs.
 *
 *   Input : [g * C_per_g][H][W]
 *   (no weight tensor)
 *   Output: [g * C_per_g][H][W]    (same size, different channel order)
 *
 * FLOPs: 0  (pure data movement / reshape + transpose)
 *
 * Shuffle rule: reshape to [g][C_per_g][H][W], transpose to [C_per_g][g][H][W],
 * flatten first two dims back. Result: channel 0 of group 0, channel 0 of
 * group 1, ..., channel 1 of group 0, channel 1 of group 1, ...
 *
 * DEMO: 4 channels from 2 groups (g0: ch0=1,ch1=2 ; g1: ch2=10,ch3=20).
 * After shuffle the order is 1,10,2,20 -- groups interleaved.
 *
 * Build:  gcc 24_channel_shuffle.c -o 24_channel_shuffle -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s  [C=%d H=%d W=%d]\n", title, C, H, W);
    for (int c = 0; c < C; c++) {
        if (C > 1) {
            printf("  channel %d:\n", c);
        }
        for (int h = 0; h < H; h++) {
            printf("   ");
            for (int w = 0; w < W; w++) {
                printf("%6.1f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

/*
 * channel_shuffle: reshape [g * n][H][W] -> [g][n][H][W] -> [n][g][H][W] -> [g*n][H][W]
 * Resulting channel order: ch[0], ch[n], ch[2n], ..., ch[1], ch[n+1], ch[2n+1], ...
 */
void channel_shuffle(const float *in, float *out, int g, int C_per_g, int H, int W)
{
    /* pure reindex: no arithmetic, no kernel -- just a logical transpose
       of the [group][channel_within_group] axes so the next grouped conv
       sees cross-group inputs without any weight overhead */
    int C = g * C_per_g; /* total channels */
    for (int i = 0; i < C * H * W; i++) {
        out[i] = 0.0f;
    }

    for (int grp = 0; grp < g; grp++) { /* grp: source group index */
        for (int ch = 0; ch < C_per_g; ch++) { /* ch: channel-within-group index */

            int in_ch  = grp * C_per_g + ch;  /* original channel index: groups are contiguous */
            int out_ch = ch  * g + grp;        /* new interleaved index: channels-first ordering */

            for (int h = 0; h < H; h++) { /* h: spatial row */
                for (int w = 0; w < W; w++) { /* w: spatial col */
                    /* copy pixel from source channel to interleaved destination channel */
                    out[out_ch * H * W + h * W + w] =
                        in[in_ch  * H * W + h * W + w];
                }
            }
        }
    }
}

int main(void)
{
    int g = 2, C_per_g = 2, H = 2, W = 2;
    int C = g * C_per_g;               /* = 4 */

    float *in  = calloc(C * H * W, sizeof(float));
    float *out = calloc(C * H * W, sizeof(float));

    /* group 0: ch0 = 1.0,  ch1 = 2.0 */
    /* group 1: ch2 = 10.0, ch3 = 20.0 */
    for (int i = 0; i < H*W; i++) { in[0*H*W+i]=1.0f;  in[1*H*W+i]=2.0f; }
    for (int i = 0; i < H*W; i++) { in[2*H*W+i]=10.0f; in[3*H*W+i]=20.0f; }

    show("INPUT  (ch0=1, ch1=2, ch2=10, ch3=20 -- 2 groups)", in, C, H, W);
    channel_shuffle(in, out, g, C_per_g, H, W);
    show("OUTPUT (ch0=1, ch1=10, ch2=2, ch3=20 -- interleaved)", out, C, H, W);

    printf("[24] check: out_ch0=%.1f (exp 1), out_ch1=%.1f (exp 10), "
           "out_ch2=%.1f (exp 2), out_ch3=%.1f (exp 20)\n",
           out[0], out[H*W], out[2*H*W], out[3*H*W]);

    free(in);
    free(out);
    return 0;
}
