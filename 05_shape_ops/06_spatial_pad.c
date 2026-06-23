/*
 * 06_spatial_pad.c
 *
 * WHAT CHANGES:  Spatial dimensions grow — [C][H][W] → [C][H+2P][W+2P].
 *                A border of P zeros is added on all four sides.
 * WHAT STAYS:    Channel count and all original pixel values (in their new positions).
 *
 * WHERE IN MODELS:
 *   - Conv layers with padding=P implicitly do this before the convolution
 *     to keep spatial dimensions constant. Here it is made explicit.
 *   - ONNX Pad op: inserted before conv when converting from PyTorch
 *     (which bundles padding) to ONNX (which separates it into a Pad node).
 *   - ViT patch embedding: input images are padded to multiples of the
 *     patch size P so the patch grid divides evenly.
 *   - Edge preprocessors: cameras sometimes pad to power-of-2 sizes before
 *     feeding NPU kernels that require aligned dimensions.
 *
 * DEMO:
 *   Input  [C=1][H=3][W=3] with ramp values 1..9.
 *   Pad=1  → output [C=1][H=5][W=5] with a zero border.
 *
 * Build:
 *   gcc -O2 -o 06_spatial_pad 06_spatial_pad.c && ./06_spatial_pad
 */

#include <stdio.h>

#define C   1
#define H   3
#define W   3
#define PAD 1
#define OH  (H + 2 * PAD)
#define OW  (W + 2 * PAD)

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
 * spatial_pad — zero-pad H and W by PAD on each side.
 *
 * Output shape: [C][OH][OW] where OH = H+2*PAD, OW = W+2*PAD.
 *
 * For each output position (oh, ow):
 *   ih = oh - PAD  (the corresponding input row)
 *   iw = ow - PAD  (the corresponding input column)
 *
 * WHY PAD is subtracted:
 *   The output grid is shifted outward by PAD positions.
 *   Row 0 of the output is the top pad row (no input).
 *   Row PAD of the output corresponds to row 0 of the input.
 *   So: ih = oh - PAD maps output coords back to input coords.
 *
 * If ih or iw is out of range [0,H) or [0,W), we are in the padding
 * border — write 0.  Otherwise copy from input.
 */
static void spatial_pad(const float *in, float *out)
{
    int c, oh, ow;

    /* Zero-fill the entire output first */
    for (c = 0; c < C * OH * OW; c++)
    {
        out[c] = 0.0f;
    }

    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < OH; oh++)
        {
            for (ow = 0; ow < OW; ow++)
            {
                int ih = oh - PAD; /* map output row to input row */
                int iw = ow - PAD; /* map output col to input col */

                /* Only copy when inside the valid input region */
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                {
                    out[c * OH * OW + oh * OW + ow] = in[c * H * W + ih * W + iw];
                }
                /* else: remains 0 from the zero-fill above */
            }
        }
    }
}

int main(void)
{
    float in[C * H * W];
    float out[C * OH * OW];
    int i;

    /* Ramp 1..9 */
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)(i + 1);
    }

    printf("=== 06_spatial_pad ===\n\n");
    show_tensor(in, C, H, W, "INPUT");

    spatial_pad(in, out);

    printf("\n");
    show_tensor(out, C, OH, OW, "OUTPUT (pad=1, zero border visible)");

    printf("\nCHECK: out[c=0][oh=%d][ow=%d]=%.1f  should equal  in[c=0][h=0][w=0]=%.1f\n",
           PAD, PAD, out[0 * OH * OW + PAD * OW + PAD], in[0 * H * W + 0 * W + 0]);

    return 0;
}
