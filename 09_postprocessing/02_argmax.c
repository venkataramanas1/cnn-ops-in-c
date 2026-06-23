/* ============================================================================
 * [02] Argmax over class axis  --  classification / segmentation decode
 * ----------------------------------------------------------------------------
 * argmax along class axis: for each (n,t) token, pick the class with the
 * highest logit. This IS the greedy decoding step in all VLA autoregressive
 * action generation.
 *
 *   Input : [N][T][C]  logits  (batch × tokens × classes)
 *   Output: [N][T]     class indices  (argmax along C)
 *
 *   4D use: [N][H][W][C] segmentation map -> [N][H][W] class-per-pixel
 *
 * DYNAMIC SHAPES: C (num_classes) is a runtime variable; memory is heap-allocated.
 *   Stride computation: element [n][t][c] = n*(T*C) + t*C + c
 *
 * MODELS: Final step of ViT, EfficientNet classification; semantic segmentation
 *   heads (DeepLab, SegFormer); VLA action token selection (argmax over vocab
 *   to get action token ID in RT-2, OpenVLA, pi0).
 *
 * Demo: [N=2][T=3][C=4] logits. Show logit grid; show argmax result [2][3].
 *   Check: known maximum positions match expected indices.
 *
 * Build: gcc 02_argmax.c -o 02_argmax -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    /* runtime shape variables (dynamic) */
    int N = 2;  /* batch size */
    int T = 3;  /* sequence length / spatial tokens */
    int C = 4;  /* number of classes -- runtime variable */

    /* heap-allocate logits [N][T][C] and output [N][T] */
    float *logits = (float *)malloc((size_t)N * T * C * sizeof(float));
    int   *out    = (int   *)malloc((size_t)N * T     * sizeof(int));

    /* --- fill demo logits: easy-to-verify maxima ---
       batch 0:
         token 0: [ 1.0, 3.0, 2.0, 0.5]  -> argmax = 1
         token 1: [ 0.2, 0.1, 4.5, 0.3]  -> argmax = 2
         token 2: [ 2.1, 1.0, 0.8, 5.0]  -> argmax = 3
       batch 1:
         token 0: [ 6.0, 1.0, 2.0, 3.0]  -> argmax = 0
         token 1: [ 0.5, 0.5, 0.5, 2.9]  -> argmax = 3
         token 2: [ 1.0, 3.7, 0.1, 0.2]  -> argmax = 1  */
    float demo[2][3][4] = {
        {{ 1.0f,  3.0f,  2.0f,  0.5f},
         { 0.2f,  0.1f,  4.5f,  0.3f},
         { 2.1f,  1.0f,  0.8f,  5.0f}},
        {{ 6.0f,  1.0f,  2.0f,  3.0f},
         { 0.5f,  0.5f,  0.5f,  2.9f},
         { 1.0f,  3.7f,  0.1f,  0.2f}}
    };

    /* copy into flat heap array using stride formula [n][t][c] = n*T*C + t*C + c */
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                logits[n*T*C + t*C + c] = demo[n][t][c];  /* row-major NTC layout */
            }
        }
    }

    /* --- print input logit grid --- */
    printf("[02] Argmax demo: logits [N=%d][T=%d][C=%d]\n", N, T, C);
    for (int n = 0; n < N; n++) {
        printf("  batch %d:\n", n);
        for (int t = 0; t < T; t++) {
            printf("    token %d: [", t);
            for (int c = 0; c < C; c++) {
                printf("%5.1f", logits[n*T*C + t*C + c]);  /* stride: n*T*C + t*C + c */
                if (c < C-1) { printf(","); }
            }
            printf("]\n");
        }
    }
    printf("\n");

    /* --- argmax along C axis ---
       For each (n,t) pair, scan all C classes and track the index of the maximum.
       No exp/normalize needed: argmax is purely an index comparison. */
    for (int n = 0; n < N; n++) {             /* n: batch index */
        for (int t = 0; t < T; t++) {         /* t: token/position index */
            int   best_c   = 0;               /* index of current best class */
            float best_val = logits[n*T*C + t*C + 0];  /* value at class 0 */

            for (int c = 1; c < C; c++) {     /* c: class index (scan from 1) */
                float val = logits[n*T*C + t*C + c];  /* logit[n,t,c] */
                if (val > best_val) {
                    best_val = val;           /* update running maximum */
                    best_c   = c;             /* record winning class index */
                }
            }
            out[n*T + t] = best_c;            /* output[n,t] = argmax_c(logits[n,t,:]) */
        }
    }

    /* --- print argmax output [N][T] --- */
    printf("[02] argmax output [N=%d][T=%d] (class indices):\n", N, T);
    for (int n = 0; n < N; n++) {
        printf("  batch %d: [", n);
        for (int t = 0; t < T; t++) {
            printf(" %d", out[n*T + t]);  /* flat index: n*T + t */
        }
        printf(" ]\n");
    }
    printf("\n");

    /* --- also demonstrate simple 1D argmax (simplest case) --- */
    float vec1d[] = {0.1f, 0.5f, 0.9f, 0.3f, 0.7f};
    int   len1d   = 5;
    int   am1d    = 0;
    for (int i = 1; i < len1d; i++) {
        if (vec1d[i] > vec1d[am1d]) { am1d = i; }  /* track running max index */
    }
    printf("[02] 1D argmax of [0.1, 0.5, 0.9, 0.3, 0.7] -> index %d (expected 2)\n\n", am1d);

    /* --- checks --- */
    int expected[2][3] = {{1, 2, 3}, {0, 3, 1}};  /* ground truth from demo data */
    int all_ok = 1;
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (out[n*T+t] != expected[n][t]) { all_ok = 0; }
        }
    }
    printf("[02] check: all argmax indices match expected = %s\n", all_ok ? "YES" : "NO");

    free(logits); free(out);
    return 0;
}
