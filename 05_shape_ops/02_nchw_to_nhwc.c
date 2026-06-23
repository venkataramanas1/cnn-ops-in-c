/*
 * 02_nchw_to_nhwc.c
 *
 * WHAT CHANGES:  Axis order — [N][C][H][W] → [N][H][W][C].
 *                Data IS copied/moved; strides change fundamentally.
 * WHAT STAYS:    Every pixel value; only its position in memory changes.
 *
 * WHERE IN MODELS:
 *   - This IS the Transpose node ONNX Runtime inserts automatically when
 *     targeting mobile execution providers (ARM NEON, Mali GPU, Qualcomm HTP).
 *   - TFLite is natively NHWC; CoreML is natively NHWC.
 *   - All PyTorch mobile vision models export with this Transpose when
 *     converted via torch.onnx.export() for edge deployment.
 *
 * WHY THE HARDWARE CARES:
 *   ARM NEON loads 4 or 8 floats in one SIMD register.
 *   In NHWC the C values for one spatial position are adjacent in memory,
 *   so a single vld1q_f32 loads all channels at once — ideal for depthwise
 *   or pointwise convolutions.  NCHW puts them C*H*W apart — unloadable
 *   in one SIMD op.
 *
 * DEMO:  N=1 C=2 H=2 W=3 (12 values).
 *   Print NCHW layout then NHWC layout — same 12 numbers, different order.
 *
 * Build:
 *   gcc -O2 -o 02_nchw_to_nhwc 02_nchw_to_nhwc.c && ./02_nchw_to_nhwc
 */

#include <stdio.h>

#define N 1
#define C 2
#define H 2
#define W 3

static void show_nchw(const float *t)
{
    int n, c, h, w;
    for (n = 0; n < N; n++)
    {
        printf("  n=%d:\n", n);
        for (c = 0; c < C; c++)
        {
            printf("    c=%d: ", c);
            for (h = 0; h < H; h++)
            {
                for (w = 0; w < W; w++)
                {
                    /* NCHW flat index: n*C*H*W + c*H*W + h*W + w */
                    printf("%5.1f", t[n * C * H * W + c * H * W + h * W + w]);
                }
                printf("  |  ");
            }
            printf("\n");
        }
    }
}

static void show_nhwc(const float *t)
{
    int n, h, w, c;
    for (n = 0; n < N; n++)
    {
        printf("  n=%d:\n", n);
        for (h = 0; h < H; h++)
        {
            printf("    h=%d: ", h);
            for (w = 0; w < W; w++)
            {
                printf("(");
                for (c = 0; c < C; c++)
                {
                    /* NHWC flat index: n*H*W*C + h*W*C + w*C + c */
                    printf("%.1f", t[n * H * W * C + h * W * C + w * C + c]);
                    if (c < C - 1)
                    {
                        printf(",");
                    }
                }
                printf(")  ");
            }
            printf("\n");
        }
    }
}

/*
 * nchw_to_nhwc — the Transpose that ONNX inserts for edge EPs.
 *
 * NCHW source index: n*C*H*W + c*H*W + h*W + w
 *   "for this (n,c,h,w) the value lives at byte n*C*H*W+c*H*W+h*W+w"
 *
 * NHWC destination index: n*H*W*C + h*W*C + w*C + c
 *   "in the new layout the same value belongs at n*H*W*C+h*W*C+w*C+c"
 *
 * We copy by iterating every (n,c,h,w) and writing the value to its new
 * position — that physical move is what Transpose means.
 */
static void nchw_to_nhwc(const float *src, float *dst)
{
    int n, c, h, w;
    for (n = 0; n < N; n++)
    {
        for (c = 0; c < C; c++)
        {
            for (h = 0; h < H; h++)
            {
                for (w = 0; w < W; w++)
                {
                    /* Read from NCHW position */
                    float val = src[n * C * H * W + c * H * W + h * W + w];
                    /* Write to NHWC position — channels become the fast axis */
                    dst[n * H * W * C + h * W * C + w * C + c] = val;
                }
            }
        }
    }
}

int main(void)
{
    /* NCHW input: channel 0 → values 10-15, channel 1 → values 20-25 */
    float src[N * C * H * W];
    float dst[N * H * W * C];
    int c, i;
    for (c = 0; c < C; c++)
    {
        for (i = 0; i < H * W; i++)
        {
            src[c * H * W + i] = (float)((c + 1) * 10 + i);
        }
    }

    printf("=== 02_nchw_to_nhwc ===\n");
    printf("\nINPUT  [N=%d][C=%d][H=%d][W=%d] — NCHW (channels grouped together):\n",
           N, C, H, W);
    show_nchw(src);

    nchw_to_nhwc(src, dst);

    printf("\nOUTPUT [N=%d][H=%d][W=%d][C=%d] — NHWC (channels interleaved per pixel):\n",
           N, H, W, C);
    show_nhwc(dst);

    printf("\nCHECK: NCHW[n=0,c=1,h=1,w=2] = %.1f"
           "  should equal  NHWC[n=0,h=1,w=2,c=1] = %.1f\n",
           src[0 * C * H * W + 1 * H * W + 1 * W + 2],
           dst[0 * H * W * C + 1 * W * C + 2 * C + 1]);

    return 0;
}
