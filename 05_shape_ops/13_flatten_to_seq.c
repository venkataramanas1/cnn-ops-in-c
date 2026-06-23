/*
 * 13_flatten_to_seq.c
 *
 * WHAT CHANGES:  [C][H][W] → [H*W][C].
 *                Each spatial position (h,w) becomes a sequence token; channels become features.
 * WHAT STAYS:    All values; total element count C*H*W.
 *
 * WHERE IN MODELS:
 *   - ViT: after patch_embed produces [N_patches][C*P*P], a linear projection
 *     maps each token to dim D, yielding [N_patches][D].  This is fed to the
 *     Transformer encoder where each patch is a sequence position.
 *   - MobileViT local attention: flattens a local window of spatial positions
 *     into a sequence before applying self-attention within the window.
 *   - VLA cross-attention: vision tokens [H*W][C] form the "keys/values"
 *     sequence; language tokens form the "queries".  Cross-attention lets
 *     each language token attend to every spatial location in the image.
 *   - Global Average Pooling alternative: instead of pooling, some architectures
 *     attend over the full [H*W] spatial sequence (set-transformer style).
 *
 * DEMO:
 *   Input  [C=3][H=2][W=2] — 3 feature channels, 4 spatial positions.
 *   Output [4][3]           — 4 tokens, each with 3-dim feature vector.
 *
 * Build:
 *   gcc -O2 -o 13_flatten_to_seq 13_flatten_to_seq.c && ./13_flatten_to_seq
 */

#include <stdio.h>

#define C  3
#define H  2
#define W  2
#define T  (H * W)   /* number of tokens = 4 */

static void show_chw(const float *t)
{
    int c, h, w;
    printf("  [C=%d][H=%d][W=%d]:\n", C, H, W);
    for (c = 0; c < C; c++)
    {
        printf("  ch%d: ", c);
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                printf("%5.1f", t[c * H * W + h * W + w]);
            }
            printf(" | ");
        }
        printf("\n");
    }
}

static void show_seq(const float *t)
{
    int tok, c;
    printf("  [T=%d tokens][C=%d features per token]:\n", T, C);
    for (tok = 0; tok < T; tok++)
    {
        int h = tok / W; /* spatial row this token came from */
        int w = tok % W; /* spatial col this token came from */
        printf("  token %d (h=%d,w=%d): [", tok, h, w);
        for (c = 0; c < C; c++)
        {
            printf(" %5.1f", t[tok * C + c]);
        }
        printf(" ]\n");
    }
}

/*
 * flatten_to_seq — reorder from channel-first to sequence-first layout.
 *
 * Input:  in[c][h][w]  = in[c*H*W + h*W + w]   (CxHxW)
 * Output: out[t][c]    = out[t*C + c]            (TxC, T=H*W)
 *
 * Mapping: token t corresponds to spatial position (h, w) where
 *   h = t / W   (row-major linearization: t = h*W + w)
 *   w = t % W
 *
 * WHY t = h*W + w (row-major):
 *   Spatial positions are scanned left-to-right, top-to-bottom.
 *   Token 0 = top-left (h=0,w=0), token 1 = (h=0,w=1), etc.
 *   This matches numpy's default C-order flatten of the H×W grid.
 *
 * The output element out[t][c] holds the C-channel feature vector
 * for the spatial position that became token t.
 * In attention: each token attends to every other token — spatial
 * locality is lost, but global relationships are captured.
 */
static void flatten_to_seq(const float *in, float *out)
{
    int t, c;
    for (t = 0; t < T; t++)
    {
        int h = t / W; /* which input row */
        int w = t % W; /* which input col */
        for (c = 0; c < C; c++)
        {
            /* Input: channel c at spatial (h,w) */
            /* Output: token t, feature dimension c */
            out[t * C + c] = in[c * H * W + h * W + w];
        }
    }
}

int main(void)
{
    float in[C * H * W];
    float out[T * C];
    int c, i;

    /* Channel c gets values starting at (c+1)*10 */
    for (c = 0; c < C; c++)
    {
        for (i = 0; i < H * W; i++)
        {
            in[c * H * W + i] = (float)((c + 1) * 10 + i);
        }
    }

    printf("=== 13_flatten_to_seq ===\n\n");
    printf("INPUT (channel-first, spatial features separated by channel):\n");
    show_chw(in);

    flatten_to_seq(in, out);

    printf("\nOUTPUT (sequence-first, all channel features co-located per token):\n");
    show_seq(out);

    /* Verify: token 2 = (h=1,w=0); its c=0 feature should be in[c=0][h=1][w=0] */
    printf("\nCHECK: token2[c=0]=%.1f  should equal  in[c=0][h=1][w=0]=%.1f\n",
           out[2 * C + 0],
           in[0 * H * W + 1 * W + 0]);

    return 0;
}
