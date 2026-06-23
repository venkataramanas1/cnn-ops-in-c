/*
 * 12_patch_embed.c
 *
 * WHAT CHANGES:  [C][H][W] image → [N_patches][C*P*P] sequence of patch vectors.
 *                Spatial grid of P×P patches; each patch is flattened to a 1D token.
 * WHAT STAYS:    All pixel values — patch_embed is a reshape+slice, no math.
 *
 * WHERE IN MODELS:
 *   - ViT (Vision Transformer): the very first op.  The image is divided into
 *     non-overlapping P×P patches; each is linearly projected to dim D (the
 *     "embedding" step, a learned weight matrix).  Here we show only the
 *     flatten/token-extraction step before the projection.
 *   - DeiT, BEiT, MAE — same patch tokenization.
 *   - Swin Transformer: patches are P=4 or P=2 and hierarchically merged.
 *   - ALL VLA vision encoders:
 *       RT-2  uses ViT-B/16 (P=16): 224×224 → 196 tokens of raw dim 768.
 *       OpenVLA uses ViT-L/14 (P=14): 336×336 → 576 tokens of raw dim 1024.
 *     patch_embed is the gateway that converts pixels to the token stream
 *     that the language model later attends to.
 *
 * DEMO:
 *   C=1, H=4, W=4, P=2 → N_patches = (4/2)*(4/2) = 4 patches.
 *   Each patch has dim = C*P*P = 4 values.
 *   Print each patch as one row of the output sequence.
 *
 * Build:
 *   gcc -O2 -o 12_patch_embed 12_patch_embed.c && ./12_patch_embed
 */

#include <stdio.h>

#define C        1
#define H        4
#define W        4
#define P        2                     /* patch size */
#define N_H      (H / P)               /* number of patch rows = 2 */
#define N_W      (W / P)               /* number of patch cols = 2 */
#define N_PATCH  (N_H * N_W)           /* total patches = 4 */
#define PATCH_DIM (C * P * P)          /* elements per patch = 4 */

static void show_image(const float *t)
{
    int c, h, w;
    printf("  [C=%d][H=%d][W=%d]:\n", C, H, W);
    for (c = 0; c < C; c++)
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

static void show_patches(const float *t)
{
    int p, d;
    printf("  [N_patches=%d][patch_dim=%d]:\n", N_PATCH, PATCH_DIM);
    for (p = 0; p < N_PATCH; p++)
    {
        int ph = p / N_W; /* patch row in the patch grid */
        int pw = p % N_W; /* patch col in the patch grid */
        printf("  patch %d (grid pos ph=%d pw=%d): [", p, ph, pw);
        for (d = 0; d < PATCH_DIM; d++)
        {
            printf(" %.1f", t[p * PATCH_DIM + d]);
        }
        printf(" ]\n");
    }
}

/*
 * patch_embed — extract and flatten non-overlapping P×P patches.
 *
 * Patch (ph, pw) covers image rows [ph*P .. ph*P+P-1]
 *                         and cols [pw*P .. pw*P+P-1].
 *
 * Within the patch, element index d = c*P*P + py*P + px, where:
 *   c  = channel index
 *   py = row offset within the patch (0..P-1)
 *   px = col offset within the patch (0..P-1)
 *
 * Corresponding image position:
 *   ih = ph*P + py   (patch's top-left row + intra-patch row)
 *   iw = pw*P + px   (patch's top-left col + intra-patch col)
 *
 * WHY ph*P + py:
 *   Patch (ph) starts at image row ph*P.  The py-th row inside the patch
 *   is ph*P + py pixels from the top of the image.  Same logic for cols.
 *
 * Output token p = ph*N_W + pw  (row-major patch grid linearization).
 */
static void patch_embed(const float *img, float *tokens)
{
    int ph, pw, c, py, px;
    for (ph = 0; ph < N_H; ph++)
    {
        for (pw = 0; pw < N_W; pw++)
        {
            int p = ph * N_W + pw; /* linear patch index */
            for (c = 0; c < C; c++)
            {
                for (py = 0; py < P; py++)
                {
                    for (px = 0; px < P; px++)
                    {
                        /* Image position */
                        int ih = ph * P + py; /* global row = patch row * P + intra-patch row */
                        int iw = pw * P + px; /* global col = patch col * P + intra-patch col */

                        /* Position within the flattened patch vector */
                        int d = c * P * P + py * P + px;

                        tokens[p * PATCH_DIM + d] = img[c * H * W + ih * W + iw];
                    }
                }
            }
        }
    }
}

int main(void)
{
    float img[C * H * W];
    float tokens[N_PATCH * PATCH_DIM];
    int i;

    /* Ramp 0..15 so pixel origin is self-evident */
    for (i = 0; i < C * H * W; i++)
    {
        img[i] = (float)i;
    }

    printf("=== 12_patch_embed ===\n\n");
    printf("INPUT image (P=%d patch size, will become %d tokens of dim %d):\n",
           P, N_PATCH, PATCH_DIM);
    show_image(img);

    patch_embed(img, tokens);

    printf("\nOUTPUT token sequence (each row = one patch flattened):\n");
    show_patches(tokens);

    /* Verification: patch 0 covers image [0:2][0:2] → values 0,1,4,5 */
    printf("\nCHECK: patch 0 = [ %.1f %.1f %.1f %.1f ] (expect 0 1 4 5 from top-left 2x2)\n",
           tokens[0 * PATCH_DIM + 0],
           tokens[0 * PATCH_DIM + 1],
           tokens[0 * PATCH_DIM + 2],
           tokens[0 * PATCH_DIM + 3]);

    return 0;
}
