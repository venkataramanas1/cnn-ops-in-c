/*
 * 08_upsample_nearest2x.c
 *
 * WHAT CHANGES:  Spatial dimensions double — [C][H][W] → [C][2H][2W].
 *                Each input pixel expands to a 2×2 block of identical values.
 * WHAT STAYS:    Channel count and all original values (replicated, not interpolated).
 *
 * WHERE IN MODELS:
 *   - FPN top-down pathway: low-resolution high-level features are nearest-
 *     upsampled 2× before being added to the same-resolution lateral features.
 *   - FCN / segmentation decoders: progressive 2× upsampling from 7×7 back
 *     to 224×224 (five 2× steps).
 *   - VLA spatial decoder (π0 uses a UNet-style decoder): upsampling in the
 *     decoder arm reconstructs full-resolution feature maps for dense prediction.
 *   - Chosen over bilinear for speed on edge devices and for training stability
 *     (no learnable parameters; gradient flows cleanly through nearest).
 *
 * DEMO:
 *   Input  [C=1][H=2][W=2] with values [1,2,3,4].
 *   Output [C=1][H=4][W=4] — each value fills a 2×2 block.
 *
 * Build:
 *   gcc -O2 -o 08_upsample_nearest2x 08_upsample_nearest2x.c && ./08_upsample_nearest2x
 */

#include <stdio.h>

#define C  1
#define H  2
#define W  2
#define R  2        /* upscale factor */
#define OH (H * R)  /* 4 */
#define OW (W * R)  /* 4 */

static void show_tensor(const float *t, int nc, int nh, int nw, const char *label)
{
    int c, h, w;
    printf("%s [C=%d][H=%d][W=%d]:\n", label, nc, nh, nw);
    for (c = 0; c < nc; c++)
    {
        printf("  channel %d:\n", c);
        for (h = 0; h < nh; h++)
        {
            printf("    ");
            for (w = 0; w < nw; w++)
            {
                printf("%5.1f", t[c * nh * nw + h * nw + w]);
            }
            printf("\n");
        }
    }
}

/*
 * upsample_nearest2x — replicate each pixel into a 2×2 output block.
 *
 * For each output position (oh, ow):
 *   ih = oh / R   (integer division — floor to the source row)
 *   iw = ow / R   (integer division — floor to the source col)
 *
 * WHY oh/R works:
 *   Output rows 0,1 both map to input row 0 (0/2=0, 1/2=0).
 *   Output rows 2,3 both map to input row 1 (2/2=1, 3/2=1).
 *   Each input row "claims" R consecutive output rows.
 *   Floor division (integer divide) does exactly this mapping.
 *
 * In ONNX: Resize node with mode="nearest", scales=[1,1,2,2].
 * In PyTorch: F.interpolate(x, scale_factor=2, mode='nearest').
 */
static void upsample_nearest2x(const float *in, float *out)
{
    int c, oh, ow;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < OH; oh++)
        {
            for (ow = 0; ow < OW; ow++)
            {
                int ih = oh / R; /* floor — nearest source row */
                int iw = ow / R; /* floor — nearest source col */
                out[c * OH * OW + oh * OW + ow] = in[c * H * W + ih * W + iw];
            }
        }
    }
}

int main(void)
{
    float in[C * H * W]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[C * OH * OW];

    printf("=== 08_upsample_nearest2x ===\n\n");
    show_tensor(in, C, H, W, "INPUT");

    upsample_nearest2x(in, out);

    printf("\n");
    show_tensor(out, C, OH, OW, "OUTPUT (nearest-neighbor 2x upsample)");

    printf("\nCHECK: out[c=0][oh=2][ow=3]=%.1f  should equal  in[c=0][ih=%d][iw=%d]=%.1f\n",
           out[0 * OH * OW + 2 * OW + 3],
           2 / R, 3 / R,
           in[0 * H * W + (2 / R) * W + (3 / R)]);

    return 0;
}
