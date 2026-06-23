/*
 * Operator  : RoI Align
 * Input     : [C][H][W] feature map + bounding box [x1,y1,x2,y2] in feature-map coords
 * Output    : [C][pool_H][pool_W] fixed-size crop via bilinear interpolation
 * FLOPs     : C * pool_H * pool_W * (4 muls + 3 adds) per sample point
 * Models    : Mask R-CNN, Faster R-CNN, VLA models that detect objects then act on them
 * NOTE      : RoI Align avoids the quantization misalignment of RoI Pool —
 *             critical for precise crop features used in detection-then-act
 *             VLA pipelines. Uses bilinear interpolation at grid points
 *             inside the ROI so no coordinates are rounded to integers.
 * DEMO      : 1 channel 6x6 ramp, box covering top-left quadrant [0,0,3,3]
 *             pool_H=pool_W=2
 * Check     : out[0][0][0] interpolated at (0.75, 0.75) in feature map
 * Build     : gcc 08_roi_align.c -O2 -o roi_align -lm
 */

#include <stdio.h>
#include <math.h>

/* print a [C][H][W] tensor as numeric grids per channel */
static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s\n", title);
    int c, h, w;
    for (c = 0; c < C; c++)
    {
        printf("  channel %d:\n", c);
        for (h = 0; h < H; h++)
        {
            printf("  ");
            for (w = 0; w < W; w++)
            {
                printf("%8.3f", m[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/*
 * bilinear_interp
 *   Sample the feature map at continuous coordinate (fy, fx).
 *   Uses 4 surrounding integer-grid neighbors weighted by distance.
 *   This is the core of RoI Align — avoids quantization snap.
 */
static float bilinear_interp(const float *feat, int H, int W,
                              float fy, float fx)
{
    /* floor and ceil of the continuous coords */
    int y0 = (int)floorf(fy);
    int x0 = (int)floorf(fx);
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    /* fractional offsets for bilinear weights */
    float dy = fy - (float)y0;
    float dx = fx - (float)x0;

    /* clamp to valid range — treat out-of-bounds as 0 */
    float v00 = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) ? feat[y0 * W + x0] : 0.0f;
    float v01 = (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W) ? feat[y0 * W + x1] : 0.0f;
    float v10 = (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W) ? feat[y1 * W + x0] : 0.0f;
    float v11 = (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W) ? feat[y1 * W + x1] : 0.0f;

    /* bilinear blend: (1-dy)(1-dx)*v00 + (1-dy)*dx*v01 + dy*(1-dx)*v10 + dy*dx*v11 */
    return (1.0f - dy) * (1.0f - dx) * v00
         + (1.0f - dy) *          dx  * v01
         +          dy  * (1.0f - dx) * v10
         +          dy  *          dx  * v11;
}

/*
 * roi_align
 *   feat    : [C][H][W] feature map
 *   out     : [C][pool_H][pool_W] output
 *   box     : {x1, y1, x2, y2} bounding box in feature-map coordinates
 *   pool_H, pool_W : output grid size
 *
 *   For each output cell (ph, pw) compute the sample point at the center
 *   of that cell's sub-region within the ROI, then bilinearly interpolate.
 *   No coordinate rounding — this is what distinguishes RoI Align from RoI Pool.
 */
static void roi_align(const float *feat, float *out,
                      int C, int H, int W,
                      float x1, float y1, float x2, float y2,
                      int pool_H, int pool_W)
{
    /* spatial extent of the ROI */
    float roi_h = y2 - y1;
    float roi_w = x2 - x1;

    /* size of each bin in feature-map units */
    float bin_h = roi_h / (float)pool_H;
    float bin_w = roi_w / (float)pool_W;

    int c, ph, pw;
    for (c = 0; c < C; c++)
    {
        const float *feat_c = feat + c * H * W;   /* pointer to this channel */
        for (ph = 0; ph < pool_H; ph++)            /* output row */
        {
            for (pw = 0; pw < pool_W; pw++)        /* output col */
            {
                /* sample point at the center of this bin */
                float fy = y1 + (ph + 0.5f) * bin_h;   /* continuous y coord in feature map */
                float fx = x1 + (pw + 0.5f) * bin_w;   /* continuous x coord in feature map */

                /* bilinear interpolation — no quantization snap */
                out[c * pool_H * pool_W + ph * pool_W + pw] =
                    bilinear_interp(feat_c, H, W, fy, fx);
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 6, W = 6;
    int pool_H = 2, pool_W = 2;
    float in[1 * 6 * 6];
    int i;

    /* ramp [1..36] */
    for (i = 0; i < C * H * W; i++)
    {
        in[i] = (float)(i + 1);
    }

    /* bounding box covering top-left quadrant [x1,y1,x2,y2] = [0,0,3,3] */
    float x1 = 0.0f, y1 = 0.0f, x2 = 3.0f, y2 = 3.0f;

    float out[1 * 2 * 2];

    show("INPUT [1][6][6] ramp 1..36:", in, C, H, W);
    roi_align(in, out, C, H, W, x1, y1, x2, y2, pool_H, pool_W);
    show("OUTPUT [1][2][2] after RoI Align (box [0,0,3,3]):", out, C, pool_H, pool_W);

    /*
     * out[0][0][0]: bin covers [y1=0..1.5, x1=0..1.5], sample at (0.75, 0.75)
     * bilinear at (0.75, 0.75):
     *   neighbors: (0,0)=1, (0,1)=2, (1,0)=7, (1,1)=8
     *   dy=0.75, dx=0.75
     *   = 0.25*0.25*1 + 0.25*0.75*2 + 0.75*0.25*7 + 0.75*0.75*8
     *   = 0.0625 + 0.375 + 1.3125 + 4.5 = 6.25
     */
    float expected = 6.25f;
    float got = out[0];
    float diff = got - expected;
    if (diff < 0.0f) diff = -diff;
    printf("CHECK: out[0][0][0] = %.4f  expected %.4f  %s\n",
           got, expected, (diff < 1e-4f) ? "PASS" : "FAIL");
    return 0;
}
