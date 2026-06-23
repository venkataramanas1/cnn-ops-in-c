/*
 * 03_nhwc_to_nchw.c
 *
 * WHAT CHANGES:  Axis order — [N][H][W][C] → [N][C][H][W].
 *                Inverse of the edge Transpose from 02_nchw_to_nhwc.c.
 * WHAT STAYS:    Every pixel value.
 *
 * WHERE IN MODELS:
 *   - Camera buffers on Android/iOS arrive in NHWC (RGBA, YUV444).
 *     Before feeding into a PyTorch NCHW model they must be converted.
 *   - TFLite inference results (NHWC) need this before post-processing
 *     code written for NCHW layouts.
 *   - Round-trip test: NCHW → NHWC → NCHW should reproduce the original.
 *
 * DEMO:
 *   Start with the same NHWC tensor produced by 02, convert back,
 *   verify values match the original NCHW source.
 *
 * Build:
 *   gcc -O2 -o 03_nhwc_to_nchw 03_nhwc_to_nchw.c && ./03_nhwc_to_nchw
 */

#include <stdio.h>

#define N 1
#define C 2
#define H 2
#define W 3

static void show_nchw(const float *t, const char *label)
{
    int n, c, h, w;
    printf("%s [N=%d][C=%d][H=%d][W=%d]:\n", label, N, C, H, W);
    for (n = 0; n < N; n++)
    {
        for (c = 0; c < C; c++)
        {
            printf("  n=%d c=%d: ", n, c);
            for (h = 0; h < H; h++)
            {
                for (w = 0; w < W; w++)
                {
                    /* NCHW flat: n*C*H*W + c*H*W + h*W + w */
                    printf("%5.1f", t[n * C * H * W + c * H * W + h * W + w]);
                }
                printf(" | ");
            }
            printf("\n");
        }
    }
}

static void show_nhwc(const float *t, const char *label)
{
    int n, h, w, c;
    printf("%s [N=%d][H=%d][W=%d][C=%d]:\n", label, N, H, W, C);
    for (n = 0; n < N; n++)
    {
        for (h = 0; h < H; h++)
        {
            printf("  n=%d h=%d: ", n, h);
            for (w = 0; w < W; w++)
            {
                printf("(");
                for (c = 0; c < C; c++)
                {
                    /* NHWC flat: n*H*W*C + h*W*C + w*C + c */
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
 * nhwc_to_nchw — inverse Transpose.
 *
 * NHWC source index: n*H*W*C + h*W*C + w*C + c
 *   "in NHWC memory, for position (n,h,w,c) the value is here"
 *
 * NCHW destination index: n*C*H*W + c*H*W + h*W + w
 *   "in NCHW memory, the same value should go here"
 *
 * The loop iterates over the NHWC logical order and scatters each
 * value to its NCHW position — the exact inverse of 02.
 */
static void nhwc_to_nchw(const float *src, float *dst)
{
    int n, h, w, c;
    for (n = 0; n < N; n++)
    {
        for (h = 0; h < H; h++)
        {
            for (w = 0; w < W; w++)
            {
                for (c = 0; c < C; c++)
                {
                    /* Read from NHWC */
                    float val = src[n * H * W * C + h * W * C + w * C + c];
                    /* Write to NCHW — channels become the slow axis */
                    dst[n * C * H * W + c * H * W + h * W + w] = val;
                }
            }
        }
    }
}

/* Helper: also build the NHWC version from NCHW for round-trip demo */
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
                    float val = src[n * C * H * W + c * H * W + h * W + w];
                    dst[n * H * W * C + h * W * C + w * C + c] = val;
                }
            }
        }
    }
}

int main(void)
{
    float nchw_orig[N * C * H * W];
    float nhwc[N * H * W * C];
    float nchw_back[N * C * H * W];
    int c, i;

    /* Build original NCHW: channel 0 → 10-15, channel 1 → 20-25 */
    for (c = 0; c < C; c++)
    {
        for (i = 0; i < H * W; i++)
        {
            nchw_orig[c * H * W + i] = (float)((c + 1) * 10 + i);
        }
    }

    printf("=== 03_nhwc_to_nchw ===\n\n");

    show_nchw(nchw_orig, "STEP 1 — Original NCHW");

    nchw_to_nhwc(nchw_orig, nhwc);
    printf("\n");
    show_nhwc(nhwc, "STEP 2 — After NCHW→NHWC (simulated camera/edge buffer)");

    nhwc_to_nchw(nhwc, nchw_back);
    printf("\n");
    show_nchw(nchw_back, "STEP 3 — After NHWC→NCHW (back to PyTorch format)");

    /* Verify round-trip */
    int ok = 1;
    int total = N * C * H * W;
    for (i = 0; i < total; i++)
    {
        if (nchw_orig[i] != nchw_back[i])
        {
            ok = 0;
        }
    }
    printf("\nCHECK: round-trip NCHW→NHWC→NCHW is %s\n", ok ? "IDENTICAL" : "WRONG");

    return 0;
}
