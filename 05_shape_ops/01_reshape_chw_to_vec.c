/*
 * 01_reshape_chw_to_vec.c
 *
 * WHAT CHANGES:  Shape only — [C][H][W] → 1D vector of length C*H*W.
 *                Data is NOT moved; the same bytes are reinterpreted.
 * WHAT STAYS:    The flat memory layout is identical in both views.
 *
 * WHERE IN MODELS:
 *   - Every classification model: the FC head flattens the final feature
 *     map before the dense projection layer.
 *   - VLA action heads (RT-2, OpenVLA): vision features are flattened and
 *     linearly projected before being fed to the language/action decoder.
 *
 * DEMO:
 *   Input  [C=2][H=2][W=3] printed as two 2×3 channel grids.
 *   Output same array printed as one row of 12 values.
 *   Every number matches its position — proving zero data movement.
 *
 * Build:
 *   gcc -O2 -o 01_reshape_chw_to_vec 01_reshape_chw_to_vec.c && ./01_reshape_chw_to_vec
 */

#include <stdio.h>

#define C 2
#define H 2
#define W 3
#define TOTAL (C * H * W)   /* 12 elements */

/* Print the tensor as channel grids (CxHxW view). */
static void show_chw(const float *t)
{
    int c, h, w;
    for (c = 0; c < C; c++)
    {
        printf("  channel %d:\n", c);
        for (h = 0; h < H; h++)
        {
            printf("    ");
            for (w = 0; w < W; w++)
            {
                /* flat index: c*H*W + h*W + w — row-major CxHxW */
                printf("%5.1f", t[c * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/* Print the tensor as a single flat row (1D vector view). */
static void show_vec(const float *t)
{
    int i;
    printf("  [ ");
    for (i = 0; i < TOTAL; i++)
    {
        /* same memory, same index — only the "shape" changed */
        printf("%.1f ", t[i]);
    }
    printf("]\n");
}

/*
 * reshape_chw_to_vec — conceptually a no-op.
 *
 * The index formula for the CxHxW view is:
 *   flat = c*H*W + h*W + w
 * The 1D vector uses the SAME flat index:
 *   flat = i
 * They are the same integer — reshape only updates the stride/shape
 * metadata (how many dimensions and their sizes); the bytes don't move.
 *
 * In PyTorch: x.reshape(-1) or x.view(C*H*W)
 * In ONNX:    Reshape node with shape=[C*H*W]
 */
static void reshape_chw_to_vec(const float *in, float *out)
{
    int i;
    for (i = 0; i < TOTAL; i++)
    {
        /* Direct copy only because C lacks a shape-metadata abstraction.
         * In a real runtime (PyTorch, TFLite) no copy happens at all. */
        out[i] = in[i];
    }
}

int main(void)
{
    /* Fill tensor with values 0..11 so position is self-documenting. */
    float tensor[TOTAL];
    int i;
    for (i = 0; i < TOTAL; i++)
    {
        tensor[i] = (float)i;
    }

    printf("=== 01_reshape_chw_to_vec ===\n");
    printf("\nINPUT  [C=%d][H=%d][W=%d] printed as channel grids:\n", C, H, W);
    show_chw(tensor);

    float vec[TOTAL];
    reshape_chw_to_vec(tensor, vec);

    printf("\nOUTPUT [length=%d] printed as 1D vector:\n", TOTAL);
    show_vec(vec);

    printf("\nCHECK: index (c=1,h=0,w=2) in CxHxW = flat %d = %.1f  |"
           "  vec[%d] = %.1f  (must match)\n",
           1 * H * W + 0 * W + 2,
           tensor[1 * H * W + 0 * W + 2],
           1 * H * W + 0 * W + 2,
           vec[1 * H * W + 0 * W + 2]);

    return 0;
}
