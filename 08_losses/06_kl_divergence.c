/* ============================================================================
 * [06] KL Divergence  --  Distribution matching
 * ----------------------------------------------------------------------------
 * KL measures how much Q diverges from P. KL=0 when P=Q exactly.
 * Asymmetric: KL(P||Q) ≠ KL(Q||P).
 * Used in VLA policy distillation: student policy Q learns from teacher P.
 *
 * Formula:
 *   KL(P||Q) = sum_i P[i] * log(P[i] / Q[i])
 *            = sum_i P[i] * (log(P[i]) - log(Q[i]))
 *
 * KL(P||Q) = 0 if and only if P = Q everywhere.
 * KL(P||Q) >= 0 always (Gibbs inequality).
 *
 * Layout: [N][C] — N probability vectors, each of length C (summing to 1)
 *   stride: element [n][c] = base + n*C + c
 *
 * Models: knowledge distillation (soft targets from teacher),
 *         VAE latent loss KL(posterior || prior),
 *         VLA policy distillation.
 *
 * Build: gcc 06_kl_divergence.c -o 06_kl_divergence -lm
 * ============================================================================ */
#include <stdio.h>
#include <math.h>

/* ---- print [N][C] probability matrix ------------------------------------- */
static void show(const char *title, const float *m, int N, int C)
{
    printf("%s  [N=%d C=%d]\n", title, N, C);
    for (int n = 0; n < N; n++) {
        printf("  row %d: ", n);
        for (int c = 0; c < C; c++) {
            printf("%8.4f", m[n * C + c]);   /* stride: row = n*C */
        }
        printf("\n");
    }
    printf("\n");
}

/* ---- KL(P||Q) for a single distribution pair [C] ------------------------- */
static float kl_single(const float *p, const float *q, int C)
{
    float kl = 0.0f;
    for (int c = 0; c < C; c++) {
        if (p[c] < 1e-10f) {
            /* P[i]=0 contributes 0 * log(...) = 0 by convention */
            continue;
        }
        /* clamp Q to avoid log(0) */
        float q_safe = q[c] < 1e-9f ? 1e-9f : q[c];

        /* P[c] * log(P[c] / Q[c]) = P[c] * (log P[c] - log Q[c]) */
        kl += p[c] * logf(p[c] / q_safe);
    }
    return kl;
}

/* ---- KL divergence over [N][C] arrays; fills per_row[N]; returns mean ---- */
static float kl_divergence(const float *P, const float *Q,
                            float *per_row, int N, int C)
{
    float total = 0.0f;
    for (int n = 0; n < N; n++) {
        /* row pointers use stride=C */
        const float *p_n = P + n * C;
        const float *q_n = Q + n * C;
        per_row[n] = kl_single(p_n, q_n, C);
        total += per_row[n];
    }
    return total / (float)N;
}

int main(void)
{
    int N = 2, C = 3;

    /* P: true distributions
     * row 0: peaked (teacher is confident about class 0)
     * row 1: near-uniform (teacher is uncertain) */
    float P[2 * 3] = {
        0.70f, 0.20f, 0.10f,   /* peaked at class 0 */
        0.33f, 0.33f, 0.34f    /* near-uniform */
    };

    /* Q: model distributions (student approximation)
     * row 0: close but not exact
     * row 1: less uniform, shifted toward class 0 */
    float Q[2 * 3] = {
        0.60f, 0.30f, 0.10f,   /* close to P row 0 */
        0.50f, 0.30f, 0.20f    /* diverges from uniform P row 1 */
    };

    show("P (true / teacher distribution)", P, N, C);
    show("Q (model / student distribution)", Q, N, C);

    float per_row[2];
    float mean_kl = kl_divergence(P, Q, per_row, N, C);

    printf("KL(P||Q) per row: row0=%.4f  row1=%.4f\n", per_row[0], per_row[1]);
    printf("mean KL(P||Q)   = %.4f\n\n", mean_kl);

    /* demonstrate asymmetry: KL(Q||P) != KL(P||Q) */
    float kl_qp[2];
    float mean_kl_qp = kl_divergence(Q, P, kl_qp, N, C);
    printf("KL(Q||P) per row: row0=%.4f  row1=%.4f\n", kl_qp[0], kl_qp[1]);
    printf("mean KL(Q||P)   = %.4f  (asymmetry: ≠ KL(P||Q))\n\n", mean_kl_qp);

    /* check: KL of identical distributions = 0 */
    float kl_self[2];
    kl_divergence(P, P, kl_self, N, C);
    printf("[06] check: KL(P||P) row0=%.6f  row1=%.6f  (expected 0.000000)\n\n",
           kl_self[0], kl_self[1]);

    /* check: KL(near-uniform||near-uniform-ish) should be near 0 */
    printf("[06] check: KL(near-uniform||uniform-ish) = %.4f  (expected near 0)\n\n",
           per_row[1]);

    return 0;
}
