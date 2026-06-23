/*
 * 07_spatial_crop_slice.c
 *
 * WHAT CHANGES:  Spatial dimensions shrink — extract [C][h1:h2][w1:w2]
 *                from [C][H][W], producing [C][h2-h1][w2-w1].
 * WHAT STAYS:    Channel count and values within the crop window.
 *
 * WHERE IN MODELS:
 *   - Data augmentation: random crop during training extracts a sub-window
 *     of the input image to improve spatial robustness.
 *   - ROI (Region of Interest) pipelines: Faster R-CNN crops features for
 *     each detected bounding box before the box classifier.
 *   - VLA task-conditioned attention: crop a spatial region of the visual
 *     feature map corresponding to the object the policy is acting on.
 *   - Center crop at inference: classify from the center 224×224 of a
 *     256×256 image without distortion.
 *
 * DEMO:
 *   Input  [C=1][H=5][W=5] with ramp values.
 *   Crop   rows [1:4], cols [1:4] → output [C=1][H=3][W=3].
 *
 * Build:
 *   gcc -O2 -o 07_spatial_crop_slice 07_spatial_crop_slice.c && ./07_spatial_crop_slice
 */

#include <stdio.h>

#define C   1
#define H   5
#define W   5

/* Crop window: rows 1..3 (exclusive end 4), cols 1..3 */
#define H1  1
#define H2  4
#define W1  1
#define W2  4

#define OH  (H2 - H1)   /* 3 */
#define OW  (W2 - W1)   /* 3 */

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
 * spatial_crop — copy a rectangular sub-region.
 *
 * For each output position (oh, ow) in [0,OH) × [0,OW):
 *   ih = oh + H1   — shift the output row back to the input coordinate
 *   iw = ow + W1   — shift the output col back to the input coordinate
 *
 * WHY H1 is added:
 *   The crop starts at input row H1.  Output row 0 corresponds to input
 *   row H1, so ih = oh + H1.  Similarly iw = ow + W1.
 *
 * In numpy: out = in[:, H1:H2, W1:W2]
 * In ONNX:  Slice node with starts=[H1,W1], ends=[H2,W2], axes=[1,2]
 */
static void spatial_crop(const float *in, float *out)
{
    int c, oh, ow;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < OH; oh++)
        {
            for (ow = 0; ow < OW; ow++)
            {
                int ih = oh + H1; /* output row oh → input row H1+oh */
                int iw = ow + W1; /* output col ow → input col W1+ow */
                out[c * OH * OW + oh * OW + ow] = in[c * H * W + ih * W + iw];
            }
        }
    }
}

int main(void)
{
    float in[C * H * W];
    float out[C * OH * OW];
    int i;

    /* Ramp 0..24 so row/col position is readable */
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)i;
    }

    printf("=== 07_spatial_crop_slice ===\n\n");
    show_tensor(in, C, H, W, "INPUT");
    printf("  Crop window: rows [%d:%d], cols [%d:%d]\n", H1, H2, W1, W2);

    spatial_crop(in, out);

    printf("\n");
    show_tensor(out, C, OH, OW, "OUTPUT (cropped sub-region)");

    /* Corner check: out[0][0][0] should be in[H1][W1] */
    printf("\nCHECK: out[0][0][0]=%.1f  should equal  in[%d][%d]=%.1f\n",
           out[0],
           H1, W1,
           in[0 * H * W + H1 * W + W1]);

    return 0;
}
