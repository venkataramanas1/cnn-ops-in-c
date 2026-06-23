/* ============================================================================
 * [03] Top-K selection  --  beam search / nucleus sampling building block
 * ----------------------------------------------------------------------------
 * Top-K is nucleus sampling's building block: keep only the K most likely
 * tokens, renormalize, sample. Dynamic K lets you trade diversity vs quality
 * at inference time.
 *
 *   Input : flat array of N values
 *   Output: K values + K indices (the K largest, in descending order)
 *
 *   Algorithm: selection-based partial sort O(N*K)
 *     Outer loop: K passes. Each pass finds the current maximum among
 *     unselected elements and appends it to the result.
 *
 * DYNAMIC SHAPES: K is a runtime parameter. Beam search uses dynamic K
 *   (beam width can vary per decode step). Memory for topk_vals/topk_idx
 *   is heap-allocated to K.
 *
 * MODELS: Beam search in VLA language decoders (top-K sampling in RT-2,
 *   OpenVLA, pi0); similarity retrieval; token selection in speculative
 *   decoding. K=1 reduces to argmax.
 *
 * Demo: N=8 values, K=3.
 *   Show all values, show top-3 values and indices.
 *   Check: top-1 index is the global argmax.
 *
 * Build: gcc 03_topk.c -o 03_topk -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

int main(void)
{
    /* runtime variables */
    int N = 8;  /* total number of values */
    int K = 3;  /* number of top values to select -- dynamic runtime parameter */

    /* heap-allocate inputs and outputs */
    float *vals     = (float *)malloc((size_t)N * sizeof(float));  /* input values */
    float *topk_val = (float *)malloc((size_t)K * sizeof(float));  /* top-K values */
    int   *topk_idx = (int   *)malloc((size_t)K * sizeof(int));    /* top-K indices */
    int   *used     = (int   *)calloc((size_t)N,  sizeof(int));    /* boolean: already selected */

    /* demo values: global maximum is at index 4 (value=9.1) */
    vals[0] = 3.2f;
    vals[1] = 7.5f;
    vals[2] = 1.0f;
    vals[3] = 6.8f;
    vals[4] = 9.1f;  /* <-- global max */
    vals[5] = 2.3f;
    vals[6] = 8.4f;  /* <-- 2nd largest */
    vals[7] = 4.6f;  /* <-- 3rd largest */

    /* --- print input --- */
    printf("[03] Top-K selection: N=%d values, K=%d\n", N, K);
    printf("  input values:\n  ");
    for (int i = 0; i < N; i++) {
        printf("  [%d]=%.1f", i, vals[i]);
    }
    printf("\n\n");

    /* --- top-K selection: K passes of linear scan ---
       Each pass finds the maximum over unselected elements.
       Marks selected elements in used[] to skip them on subsequent passes.
       Time: O(N*K) -- acceptable for small K (beam width typically 1-5). */
    for (int k = 0; k < K; k++) {               /* k: which top-k slot to fill */
        int   best_idx = -1;
        float best_val = -FLT_MAX;              /* start below all real values */

        for (int i = 0; i < N; i++) {           /* i: scan all N candidates */
            if (used[i]) { continue; }          /* skip already-selected elements */
            if (vals[i] > best_val) {
                best_val = vals[i];             /* new running maximum */
                best_idx = i;                   /* track its index */
            }
        }

        topk_val[k] = best_val;                 /* store k-th largest value */
        topk_idx[k] = best_idx;                 /* store k-th largest index */
        used[best_idx] = 1;                     /* mark as used so next pass skips it */
    }

    /* --- print results --- */
    printf("[03] Top-%d results (descending order):\n", K);
    for (int k = 0; k < K; k++) {
        printf("  rank %d: index=%d  value=%.1f\n", k+1, topk_idx[k], topk_val[k]);
    }
    printf("\n");

    /* --- checks --- */
    /* top-1 index should be 4 (value 9.1, the global max) */
    printf("[03] check: top-1 index = %d (expected 4, global argmax)\n", topk_idx[0]);

    /* verify descending order */
    int order_ok = 1;
    for (int k = 0; k < K-1; k++) {
        if (topk_val[k] < topk_val[k+1]) { order_ok = 0; }  /* must be non-increasing */
    }
    printf("[03] check: values in descending order = %s\n", order_ok ? "YES" : "NO");

    /* expected: indices 4 (9.1), 6 (8.4), 1 (7.5) */
    int exp_idx[3] = {4, 6, 1};
    int idx_ok = (topk_idx[0]==exp_idx[0] && topk_idx[1]==exp_idx[1] && topk_idx[2]==exp_idx[2]);
    printf("[03] check: top-3 indices match {4,6,1} = %s\n", idx_ok ? "YES" : "NO");

    free(vals); free(topk_val); free(topk_idx); free(used);
    return 0;
}
