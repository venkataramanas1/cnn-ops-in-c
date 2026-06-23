/*
 * 14_squeeze_unsqueeze.c
 *
 * WHAT CHANGES:  Rank (number of dimensions) only.
 *                Squeeze: remove a dimension of size 1.  [1][H][W] → [H][W].
 *                Unsqueeze: add a dimension of size 1.   [H][W] → [1][H][W].
 *                Data is NOT moved.  Same flat index, different shape metadata.
 * WHAT STAYS:    Every value and the total element count.
 *
 * WHERE IN MODELS:
 *   - Attention bias addition: attention scores are [B][heads][T][T]; a bias
 *     of shape [1][1][T][T] must be unsqueezed to broadcast over B and heads.
 *   - ONNX exports: ONNX frequently inserts Squeeze/Unsqueeze nodes between
 *     ops with mismatched rank (e.g., after a Reduce op that collapses a dim,
 *     or before a broadcast-heavy op that needs explicit batch dim).
 *   - Batch dimension: a single image [C][H][W] is unsqueezed to [1][C][H][W]
 *     before feeding a batched model.
 *
 * DEMO:
 *   Start with [1][3][3] (rank 3, batch size 1).
 *   Squeeze the leading 1 → [3][3] (rank 2).
 *   Unsqueeze back → [1][3][3] (rank 3).
 *   Same 9 values throughout; print all three views.
 *
 * Build:
 *   gcc -O2 -o 14_squeeze_unsqueeze 14_squeeze_unsqueeze.c && ./14_squeeze_unsqueeze
 */

#include <stdio.h>

#define N  1   /* the size-1 batch dimension */
#define H  3
#define W  3
#define TOTAL (N * H * W)   /* 9 elements */

/* Print as [N][H][W] — rank-3 view */
static void show_rank3(const float *t, const char *label)
{
    int n, h, w;
    printf("%s [N=%d][H=%d][W=%d]:\n", label, N, H, W);
    for (n = 0; n < N; n++)
    {
        printf("  n=%d:\n", n);
        for (h = 0; h < H; h++)
        {
            printf("    ");
            for (w = 0; w < W; w++)
            {
                /* flat: n*H*W + h*W + w — same formula whether N=1 or not */
                printf("%5.1f", t[n * H * W + h * W + w]);
            }
            printf("\n");
        }
    }
}

/* Print as [H][W] — rank-2 view of the SAME flat array */
static void show_rank2(const float *t, const char *label)
{
    int h, w;
    printf("%s [H=%d][W=%d]:\n", label, H, W);
    for (h = 0; h < H; h++)
    {
        printf("  ");
        for (w = 0; w < W; w++)
        {
            /* flat: h*W + w — N=1 so n*H*W vanishes; index is identical */
            printf("%5.1f", t[h * W + w]);
        }
        printf("\n");
    }
}

/*
 * squeeze — conceptually a no-op; only the shape descriptor changes.
 *
 * [N=1][H][W] flat index: n*H*W + h*W + w = 0*H*W + h*W + w = h*W + w
 * [H][W]      flat index: h*W + w
 *
 * WHY the indices are identical:
 *   When N=1 the leading dimension contributes 0 to any flat index (n is
 *   always 0, so n*H*W = 0).  Removing that dimension leaves the offset
 *   formula h*W+w, which is exactly what the rank-2 indexing uses.
 *   No bytes move; we just stop telling the runtime the first axis exists.
 *
 * In C we do a trivial memcpy to make the demo explicit.
 * In PyTorch: x.squeeze(0)   or  x.squeeze(dim=0)
 * In ONNX:    Squeeze node with axes=[0]
 */
static void squeeze(const float *in, float *out)
{
    int i;
    for (i = 0; i < TOTAL; i++)
    {
        /* Index is unchanged — prove squeeze is a metadata-only op */
        out[i] = in[i];
    }
}

/*
 * unsqueeze — inverse no-op; same argument applies in reverse.
 *
 * [H][W]      flat: h*W + w
 * [N=1][H][W] flat: 0*H*W + h*W + w = h*W + w  (n is always 0)
 *
 * In PyTorch: x.unsqueeze(0)
 * In ONNX:    Unsqueeze node with axes=[0]
 */
static void unsqueeze(const float *in, float *out)
{
    int i;
    for (i = 0; i < TOTAL; i++)
    {
        out[i] = in[i];
    }
}

int main(void)
{
    float rank3[TOTAL];   /* [1][3][3] */
    float rank2[TOTAL];   /* [3][3]    — same bytes */
    float rank3b[TOTAL];  /* [1][3][3] — round-trip */
    int i;

    /* Ramp 1..9 */
    for (i = 0; i < TOTAL; i++)
    {
        rank3[i] = (float)(i + 1);
    }

    printf("=== 14_squeeze_unsqueeze ===\n\n");
    show_rank3(rank3, "ORIGINAL (rank-3 view)");

    squeeze(rank3, rank2);
    printf("\n");
    show_rank2(rank2, "AFTER squeeze (rank-2 view, same 9 bytes)");

    unsqueeze(rank2, rank3b);
    printf("\n");
    show_rank3(rank3b, "AFTER unsqueeze (rank-3 view restored)");

    /* Verify: all three arrays have identical flat values */
    int ok = 1;
    for (i = 0; i < TOTAL; i++)
    {
        if (rank3[i] != rank2[i] || rank2[i] != rank3b[i])
        {
            ok = 0;
        }
    }
    printf("\nCHECK: all three arrays are %s (squeeze/unsqueeze are no-ops on data)\n",
           ok ? "IDENTICAL" : "DIFFERENT (bug)");

    return 0;
}
