/*
 * 09_upsample_bilinear2x.c
 *
 * WHAT CHANGES:  Spatial dimensions double — [C][H][W] → [C][2H][2W].
 *                Output values are smoothly interpolated from 4 neighboring input pixels.
 * WHAT STAYS:    Channel count.
 *
 * WHERE IN MODELS:
 *   - DeepLab v3+: the decoder path uses bilinear upsampling ×4 and ×8
 *     for smooth semantic segmentation boundaries.
 *   - FPN variants: some implementations use bilinear instead of nearest
 *     for cleaner feature alignment.
 *   - VLA policy decoders: bilinear gives smoother spatial feature maps
 *     for dense visuomotor prediction (end-effector position maps).
 *
 * HOW BILINEAR WORKS:
 *   For output pixel (oh, ow), find the fractional source coordinate:
 *     fx = (ow + 0.5) / R - 0.5   (align_corners=False convention)
 *     fy = (oh + 0.5) / R - 0.5
 *   The four surrounding input pixels are:
 *     (x0,y0) = (floor(fx), floor(fy))   (top-left)
 *     (x1,y1) = (x0+1, y0+1)             (bottom-right)
 *   Blend weights:
 *     dx = fx - x0  (fractional distance from x0)
 *     dy = fy - y0
 *   Output = (1-dy)*(1-dx)*in[y0,x0] + (1-dy)*dx*in[y0,x1]
 *           +    dy*(1-dx)*in[y1,x0] +    dy*dx*in[y1,x1]
 *
 * DEMO:
 *   Input  [C=1][H=2][W=2] values [1,2,3,4].
 *   Output [C=1][H=4][W=4] — compare with nearest (file 08) to see smoothing.
 *
 * Build:
 *   gcc -O2 -o 09_upsample_bilinear2x 09_upsample_bilinear2x.c -lm && ./09_upsample_bilinear2x
 */

#include <stdio.h>
#include <math.h>

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
                printf("%6.3f", t[c * nh * nw + h * nw + w]);
            }
            printf("\n");
        }
    }
}

/* Clamp helper: keep index within [0, max-1] */
static int clamp(int v, int lo, int hi)
{
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

/*
 * upsample_bilinear2x — 4-corner weighted blend per output pixel.
 *
 * align_corners=False (PyTorch / ONNX default):
 *   The pixel is treated as a unit square centered at (i+0.5).
 *   Source coordinate: fx = (ow + 0.5) / R - 0.5
 *
 * WHY the +0.5/-0.5 convention:
 *   Without it, output corner (0,0) maps to input corner (0,0) exactly
 *   (align_corners=True) — this distorts the image when H or W is small.
 *   The half-pixel shift ensures edge output pixels sample half-way inside
 *   the boundary input pixel, matching how image resizing is defined in
 *   most imaging standards.
 */
static void upsample_bilinear2x(const float *in, float *out)
{
    int c, oh, ow;
    for (c = 0; c < C; c++)
    {
        for (oh = 0; oh < OH; oh++)
        {
            for (ow = 0; ow < OW; ow++)
            {
                /* Map output pixel center to fractional input coordinate */
                float fy = (oh + 0.5f) / (float)R - 0.5f;
                float fx = (ow + 0.5f) / (float)R - 0.5f;

                /* Top-left corner of the 2×2 input neighborhood */
                int y0 = (int)floorf(fy);
                int x0 = (int)floorf(fx);

                /* Bottom-right corner */
                int y1 = y0 + 1;
                int x1 = x0 + 1;

                /* Fractional distances toward bottom-right */
                float dy = fy - (float)y0;
                float dx = fx - (float)x0;

                /* Clamp to valid input bounds (border replication) */
                y0 = clamp(y0, 0, H - 1);
                x0 = clamp(x0, 0, W - 1);
                y1 = clamp(y1, 0, H - 1);
                x1 = clamp(x1, 0, W - 1);

                /* Bilinear blend: weighted sum of 4 corners */
                float tl = in[c * H * W + y0 * W + x0]; /* top-left     weight (1-dy)(1-dx) */
                float tr = in[c * H * W + y0 * W + x1]; /* top-right    weight (1-dy)*dx    */
                float bl = in[c * H * W + y1 * W + x0]; /* bottom-left  weight dy*(1-dx)    */
                float br = in[c * H * W + y1 * W + x1]; /* bottom-right weight dy*dx        */

                out[c * OH * OW + oh * OW + ow] =
                    (1.0f - dy) * (1.0f - dx) * tl +
                    (1.0f - dy) *          dx  * tr +
                             dy * (1.0f - dx) * bl +
                             dy *          dx  * br;
            }
        }
    }
}

int main(void)
{
    float in[C * H * W]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[C * OH * OW];
    /* Nearest for comparison */
    float near[C * OH * OW];
    int oh, ow;
    for (oh = 0; oh < OH; oh++)
    {
        for (ow = 0; ow < OW; ow++)
        {
            near[oh * OW + ow] = in[(oh / R) * W + (ow / R)];
        }
    }

    printf("=== 09_upsample_bilinear2x ===\n\n");
    show_tensor(in, C, H, W, "INPUT");

    printf("\n");
    show_tensor(near, C, OH, OW, "NEAREST 2x (for comparison, hard edges)");

    upsample_bilinear2x(in, out);

    printf("\n");
    show_tensor(out, C, OH, OW, "BILINEAR 2x (smooth blend between pixels)");

    printf("\nCHECK: corners of bilinear output should approach input corners:\n");
    printf("  out[0][0]=%.3f (near in[0][0]=%.1f)\n",
           out[0 * OH * OW + 0 * OW + 0], in[0]);
    printf("  out[3][3]=%.3f (near in[1][1]=%.1f)\n",
           out[0 * OH * OW + 3 * OW + 3], in[1 * W + 1]);

    return 0;
}
