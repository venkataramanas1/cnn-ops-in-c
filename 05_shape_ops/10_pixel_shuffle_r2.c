/*
 * 10_pixel_shuffle_r2.c
 *
 * WHAT CHANGES:  [C*r*r][H][W] → [C][H*r][W*r]  with r=2.
 *                r*r sub-channel planes are rearranged into one higher-res plane.
 * WHAT STAYS:    Total element count (C*r*r*H*W unchanged).
 *
 * WHERE IN MODELS:
 *   - ESRGAN / Real-ESRGAN super-resolution: the final upsampling stage
 *     uses PixelShuffle to go from C*4 channels at low-res to C channels
 *     at 2× resolution.  Efficient — all computation happens at low-res.
 *   - Real-time image restoration (NAFNET, SPADE): same reason.
 *   - VLA visual decoders: some efficient upsampling implementations in
 *     the decoder arm prefer PixelShuffle over transposed conv because it
 *     avoids checkerboard artifacts.
 *
 * WHY PIXELSHUFFLE IS BETTER THAN TRANSPOSED CONV FOR UPSAMPLING:
 *   Transposed conv with stride=2 produces checkerboard artifacts because
 *   stride-2 interleaves padded zeros.  PixelShuffle rearranges existing
 *   values — no zeros, no checkerboard.
 *
 * DEMO:
 *   Input  [C*r*r=4][H=2][W=2] — 4 sub-channels, low-res 2×2.
 *     sub-channel 0: [10,11,12,13] — top-left pixels for output
 *     sub-channel 1: [20,21,22,23] — top-right pixels
 *     sub-channel 2: [30,31,32,33] — bottom-left pixels
 *     sub-channel 3: [40,41,42,43] — bottom-right pixels
 *   Output [C=1][H*r=4][W*r=4] — full-res 4×4.
 *
 * Build:
 *   gcc -O2 -o 10_pixel_shuffle_r2 10_pixel_shuffle_r2.c && ./10_pixel_shuffle_r2
 */

#include <stdio.h>

#define C_OUT 1   /* output channels */
#define R     2   /* upscale factor */
#define H     2   /* input height */
#define W     2   /* input width */
#define C_IN  (C_OUT * R * R)   /* 4: r*r sub-channels per output channel */
#define OH    (H * R)           /* 4 */
#define OW    (W * R)           /* 4 */

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
                printf("%5.0f", t[c * nh * nw + h * nw + w]);
            }
            printf("\n");
        }
    }
}

/*
 * pixel_shuffle_r2 — unfold r*r sub-channels into spatial positions.
 *
 * Input index for sub-channel (c*R*R + rh*R + rw) at position (h,w):
 *   in[ (c*R*R + rh*R + rw) * H * W + h * W + w ]
 *
 * Output position (oh, ow) = (h*R + rh, w*R + rw):
 *   out[ c * OH * OW + oh * OW + ow ]
 *
 * WHY this works:
 *   Think of the r*r sub-channels as a tile of r×r "sub-pixels".
 *   Sub-channel index (rh*R + rw) encodes which position within the 2×2
 *   tile this sub-channel fills:
 *     rh=0,rw=0 → top-left  of each 2×2 output block
 *     rh=0,rw=1 → top-right
 *     rh=1,rw=0 → bottom-left
 *     rh=1,rw=1 → bottom-right
 *   The output pixel at (h*R+rh, w*R+rw) picks up the value from input
 *   spatial position (h,w) in sub-channel (rh*R+rw).
 *
 * In PyTorch: torch.nn.PixelShuffle(upscale_factor=2)
 * In ONNX:    DepthToSpace node with blocksize=2
 */
static void pixel_shuffle_r2(const float *in, float *out)
{
    int c, h, w, rh, rw;
    for (c = 0; c < C_OUT; c++)
    {
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                for (rh = 0; rh < R; rh++)
                {
                    for (rw = 0; rw < R; rw++)
                    {
                        /* Which sub-channel holds the pixel at tile offset (rh,rw)? */
                        int ic = c * R * R + rh * R + rw; /* input channel index */

                        /* Where does it go in the output? */
                        int oh = h * R + rh; /* output row: input row * R + tile row offset */
                        int ow = w * R + rw; /* output col: input col * R + tile col offset */

                        out[c * OH * OW + oh * OW + ow] = in[ic * H * W + h * W + w];
                    }
                }
            }
        }
    }
}

int main(void)
{
    float in[C_IN * H * W];
    float out[C_OUT * OH * OW];
    int ic, i;

    /* Sub-channel ic gets value (ic+1)*10 + pixel_index so we can trace origins */
    for (ic = 0; ic < C_IN; ic++)
    {
        for (i = 0; i < H * W; i++)
        {
            in[ic * H * W + i] = (float)((ic + 1) * 10 + i);
        }
    }

    printf("=== 10_pixel_shuffle_r2 ===\n\n");
    printf("INPUT: 4 sub-channels (will become 2x2 tile positions in output)\n");
    show_tensor(in, C_IN, H, W, "  sub-channels");

    pixel_shuffle_r2(in, out);

    printf("\n");
    show_tensor(out, C_OUT, OH, OW, "OUTPUT (sub-channels unfolded to 2x spatial res)");

    printf("\nCHECK: in[ic=0][h=0][w=1]=%.0f should appear at out[c=0][oh=0][ow=2] (rh=0,rw=0)\n"
           "       out[c=0][oh=0][ow=2]=%.0f\n",
           in[0 * H * W + 0 * W + 1],
           out[0 * OH * OW + 0 * OW + 2]);

    return 0;
}
