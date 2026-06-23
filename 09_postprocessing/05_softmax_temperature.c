/* ============================================================================
 * [05] Softmax with temperature scaling  --  VLA/LLM sampling control
 * ----------------------------------------------------------------------------
 * Temperature scaling is the single most important inference-time knob for
 * VLA/LLM: T<1 makes the model more deterministic (robot picks most likely
 * action), T>1 more exploratory. RT-2 paper ablates this.
 *
 *   p[i] = exp(x[i] / T) / sum_j( exp(x[j] / T) )
 *
 *   T < 1  -> sharper distribution (max prob increases, entropy decreases)
 *   T = 1  -> standard softmax
 *   T > 1  -> softer / more uniform distribution
 *
 *   Input layout: [N][seq_T][C]  -- apply per-position along C (class) axis
 *   Numerically stable: subtract max before exp to avoid overflow.
 *
 * DYNAMIC SHAPES: temperature is a runtime scalar; C is runtime.
 *   Stride: element [n][t][c] = n*(seq_T*C) + t*C + c
 *
 * MODELS: VLA action decoding temperature (RT-2 uses T=1.0 at eval, T can be
 *   tuned post-training); LLM sampling temperature; knowledge distillation
 *   uses T>1 to soften teacher logits for training the student.
 *
 * Demo: same logits [1,2,3,4,1] at T=0.5 (sharp), T=1.0 (normal), T=2.0 (soft).
 *   Show probabilities at each temperature side by side.
 *   Check: sum=1 at each T; T=0.5 max_prob > T=2.0 max_prob.
 *
 * Build: gcc 05_softmax_temperature.c -o 05_softmax_temperature -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- softmax with temperature for one position's logit vector ---
   Operates on logits[0..C-1], writes probs[0..C-1].
   Temperature T divides each logit before exp. */
static void softmax_temp(const float *logits, float *probs, int C, float T)
{
    /* find max for numerical stability: subtract before exp to prevent overflow */
    float maxv = logits[0];
    for (int c = 1; c < C; c++) {
        if (logits[c] > maxv) { maxv = logits[c]; }  /* running max over class axis */
    }

    /* compute exp(x[c]/T - max/T) = exp((x[c]-max)/T) */
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        probs[c] = expf((logits[c] - maxv) / T);  /* scaled & stabilized exp */
        sum += probs[c];                            /* accumulate partition function */
    }

    /* normalize: divide by partition function to get valid probability distribution */
    for (int c = 0; c < C; c++) {
        probs[c] /= sum;  /* probs[c] = exp(x[c]/T) / sum_j exp(x[j]/T) */
    }
}

int main(void)
{
    /* runtime shape variables */
    int N     = 1;  /* batch size */
    int seq_T = 1;  /* sequence length (one position for the demo) */
    int C     = 5;  /* number of classes / vocab tokens -- runtime */

    /* three temperatures to compare */
    float temps[3] = {0.5f, 1.0f, 2.0f};
    int   n_temps  = 3;

    /* heap-allocate [N][seq_T][C] logits and probabilities */
    float *logits = (float *)malloc((size_t)N * seq_T * C * sizeof(float));
    float *probs  = (float *)malloc((size_t)C * sizeof(float));  /* one position at a time */

    /* demo logits: same vector used at all three temperatures */
    /* [1, 2, 3, 4, 1]  --  maximum at index 3 */
    float demo_logits[5] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f};
    for (int c = 0; c < C; c++) {
        logits[0*seq_T*C + 0*C + c] = demo_logits[c];  /* fill [n=0][t=0][c] */
    }

    /* --- print input logits --- */
    printf("[05] Softmax with temperature: logits = [");
    for (int c = 0; c < C; c++) {
        printf("%.1f%s", demo_logits[c], c<C-1 ? ", " : "");
    }
    printf("]\n\n");

    /* --- compute and print softmax at each temperature --- */
    printf("  %-6s  %-50s  %-8s  %-8s\n", "T", "probabilities (C=5)", "sum", "max_prob");
    printf("  %-6s  %-50s  %-8s  %-8s\n", "------", "--------------------------------------------------", "--------", "--------");

    float max_prob_sharp = 0.0f;  /* max_prob at T=0.5 for final check */
    float max_prob_soft  = 0.0f;  /* max_prob at T=2.0 for final check */
    int   sums_ok = 1;            /* track whether all sums == 1 */

    for (int ti = 0; ti < n_temps; ti++) {
        float T = temps[ti];  /* runtime temperature scalar */

        /* get logit pointer for position [n=0][t=0] */
        const float *pos_logits = logits + 0*seq_T*C + 0*C;  /* stride: n*seq_T*C + t*C */

        softmax_temp(pos_logits, probs, C, T);  /* apply temperature softmax */

        /* compute sum and max for verification */
        float sum    = 0.0f;
        float maxp   = 0.0f;
        for (int c = 0; c < C; c++) {
            sum  += probs[c];
            if (probs[c] > maxp) { maxp = probs[c]; }
        }

        /* print row */
        printf("  T=%-4.1f  [", T);
        for (int c = 0; c < C; c++) {
            printf("%7.4f%s", probs[c], c<C-1 ? ", " : "");
        }
        printf("]  %8.6f  %8.4f\n", sum, maxp);

        if (fabsf(sum - 1.0f) > 1e-5f) { sums_ok = 0; }  /* check sum == 1 */

        if (T == 0.5f) { max_prob_sharp = maxp; }
        if (T == 2.0f) { max_prob_soft  = maxp; }
    }
    printf("\n");

    /* --- checks --- */
    printf("[05] check: all probability sums == 1.0 -> %s\n", sums_ok ? "YES" : "NO");
    printf("[05] check: T=0.5 max_prob (%.4f) > T=2.0 max_prob (%.4f) -> %s\n",
           max_prob_sharp, max_prob_soft,
           (max_prob_sharp > max_prob_soft) ? "YES (sharper at T<1)" : "NO (unexpected)");

    free(logits); free(probs);
    return 0;
}
