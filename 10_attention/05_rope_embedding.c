/* ============================================================================
 * [05] Rotary Position Embedding (RoPE)
 * ----------------------------------------------------------------------------
 * Applied to Q and K tensors before the attention dot product.
 *
 * For each pair of dims (2i, 2i+1) at position t:
 *   theta_i  = 1 / 10000^(2i / Dh)          frequency for pair i
 *   cos_t    = cos(t * theta_i)
 *   sin_t    = sin(t * theta_i)
 *   q'_{2i}   = q_{2i}   * cos_t - q_{2i+1} * sin_t   [real part of rotation]
 *   q'_{2i+1} = q_{2i}   * sin_t + q_{2i+1} * cos_t   [imag part of rotation]
 *
 * Key property: Q·K^T after RoPE depends only on RELATIVE position (t_q - t_k).
 *   Perfect for autoregressive generation where sequence grows dynamically.
 *
 * RoPE encodes absolute position as a 2D rotation in each feature pair.
 * The KEY property: when you compute Q·K^T after rotation, only the RELATIVE
 * position (t_q - t_k) matters — perfect for autoregressive generation where
 * sequence grows dynamically.
 *
 * 4D layout [N][H][T][Dh]:
 *   flat index: n*H*T*Dh + h*T*Dh + t*Dh + d
 *
 * T is dynamic — sequence length varies per sample in VLA inference.
 *   int stride_H = T * Dh;   depends on runtime T — computed at call time
 *
 * Models: LLaMA, Mistral, Gemma, RT-2 (PaLM backbone), OpenVLA.
 *   RoPE replaced learned position embeddings in ALL modern LLM-based VLA models.
 *
 * Build: gcc 05_rope_embedding.c -o 05_rope_embedding -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void print_grid(const char *title, const float *m, int rows, int cols)
{
    printf("%s  [%d x %d]\n", title, rows, cols);
    for (int r = 0; r < rows; r++) {
        printf("   ");
        for (int c = 0; c < cols; c++) {
            printf("%8.4f", m[r*cols + c]);
        }
        printf("\n");
    }
    printf("\n");
}

/* ============================================================
 * rope_apply: in-place RoPE on [N][H][T][Dh] tensor
 *   For each (n, h, t, i) pair:
 *     rotate dims (2i, 2i+1) by angle theta_i * t
 * ============================================================ */
static void rope_apply(float *x,  /* [N][H][T][Dh] — modified in place */
                       int N, int H, int T, int Dh,
                       float base)  /* base=10000.0 per original RoPE paper */
{
    /* T is dynamic — sequence length varies per sample in VLA inference */
    int stride_d   = 1;
    int stride_T   = Dh;           /* feature stride per token position */
    int stride_H   = T * Dh;       /* depends on runtime T — computed at call time */
    int stride_N   = H * T * Dh;

    int n_pairs = Dh / 2; /* number of (cos, sin) rotation pairs in Dh dims */

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int t = 0; t < T; t++) {
                /* base pointer for this (n, h, t) token vector */
                float *tok = x + n*stride_N + h*stride_H + t*stride_T;

                for (int i = 0; i < n_pairs; i++) { /* i: rotation pair index (0..Dh/2-1) */

                    /* frequency for pair i: theta_i = 1 / base^(2i / Dh)
                     * lower i → higher frequency (changes fast with t)
                     * higher i → lower frequency (changes slowly with t) */
                    float theta_i = 1.0f / powf(base, (float)(2*i) / (float)Dh);

                    /* angle for this position and pair */
                    float angle = (float)t * theta_i; /* absolute position t encoded as rotation */

                    float cos_t = cosf(angle); /* real component of rotation */
                    float sin_t = sinf(angle); /* imaginary component of rotation */

                    /* read the two feature values for this pair */
                    float q0 = tok[2*i    ]; /* q_{2i}:   first element of pair */
                    float q1 = tok[2*i + 1]; /* q_{2i+1}: second element of pair */

                    /* apply 2D rotation: equivalent to multiplying by e^{i*angle} in complex space */
                    tok[2*i    ] = q0 * cos_t - q1 * sin_t; /* rotated real part */
                    tok[2*i + 1] = q0 * sin_t + q1 * cos_t; /* rotated imag part */
                }
            }
        }
    }
}

/* ============================================================
 * Demo: N=1 H=1 T=4 Dh=4 (2 rotation pairs)
 * Q: constant [1,1,1,1] per token — position encoding visible as deviation from constant
 * ============================================================ */
int main(void)
{
    int N = 1, H = 1, T = 4, Dh = 4;
    float base = 10000.0f; /* standard RoPE base frequency */

    /* T is dynamic — strides depend on runtime T */
    int stride_H = T * Dh;  /* depends on runtime T — computed at call time */
    int stride_N = H * T * Dh;

    int sz = N * H * T * Dh;

    float *Q_before = malloc(sz * sizeof(float)); /* original Q */
    float *Q_after  = malloc(sz * sizeof(float)); /* RoPE-rotated Q */

    /* Q: constant [1,1,1,1] per token
     * position encoding is visible as the change FROM this constant */
    for (int i = 0; i < sz; i++) {
        Q_before[i] = 1.0f;
        Q_after[i]  = 1.0f;
    }

    printf("=== Rotary Position Embedding (RoPE) Demo ===\n");
    printf("N=%d H=%d T=%d Dh=%d  base=%.0f\n", N, H, T, Dh, base);
    printf("RoPE encodes absolute position as a 2D rotation in each feature pair.\n");
    printf("KEY property: Q*K^T after RoPE depends only on RELATIVE position (t_q - t_k).\n\n");

    /* show frequencies per pair */
    printf("Rotation frequencies per pair (theta_i = 1/base^(2i/Dh)):\n");
    int n_pairs = Dh / 2;
    for (int i = 0; i < n_pairs; i++) {
        float theta_i = 1.0f / powf(base, (float)(2*i) / (float)Dh);
        printf("  pair %d: theta_%d = %.6f\n", i, i, theta_i);
    }
    printf("\n");

    /* show cos/sin values at each position for pair 0 */
    printf("cos/sin at each token position for pair 0:\n");
    float theta_0 = 1.0f / powf(base, 0.0f); /* 2*0/Dh = 0 → theta = 1.0 */
    for (int t = 0; t < T; t++) {
        printf("  t=%d: cos=%.4f  sin=%.4f\n",
               t, cosf((float)t * theta_0), sinf((float)t * theta_0));
    }
    printf("\n");

    print_grid("Q BEFORE RoPE  [T x Dh]", Q_before, T, Dh);

    /* apply RoPE in place */
    rope_apply(Q_after, N, H, T, Dh, base);

    print_grid("Q AFTER RoPE   [T x Dh]", Q_after, T, Dh);

    /* show per-position change to make rotation visible */
    printf("Per-token rotation effect (after - before)  [T x Dh]:\n");
    for (int t = 0; t < T; t++) {
        printf("  t=%d: ", t);
        for (int d = 0; d < Dh; d++) {
            float diff = Q_after[t*Dh + d] - Q_before[t*Dh + d];
            printf("%8.4f", diff);
        }
        printf("\n");
    }
    printf("\n");

    /* checks */
    printf("--- Checks ---\n");

    /* at position t=0: cos(0)=1, sin(0)=0 → Q unchanged (rotation by 0) */
    int ok = 1;
    int t0_ok = 1;
    for (int d = 0; d < Dh; d++) {
        if (fabsf(Q_after[0*Dh + d] - Q_before[0*Dh + d]) > 1e-5f) {
            t0_ok = 0;
        }
    }
    printf("  t=0: cos=1 sin=0 → Q unchanged (rotation by 0): %s\n",
           t0_ok ? "PASS" : "FAIL");
    if (!t0_ok) { ok = 0; }

    /* rotation preserves vector norm (rotation is an isometry) */
    for (int t = 0; t < T; t++) {
        float norm_before = 0.0f;
        float norm_after  = 0.0f;
        for (int d = 0; d < Dh; d++) {
            norm_before += Q_before[t*Dh + d] * Q_before[t*Dh + d];
            norm_after  += Q_after[t*Dh  + d] * Q_after[t*Dh  + d];
        }
        norm_before = sqrtf(norm_before);
        norm_after  = sqrtf(norm_after);
        int norm_ok = fabsf(norm_before - norm_after) < 1e-4f;
        printf("  t=%d: norm before=%.4f after=%.4f (rotation preserves norm): %s\n",
               t, norm_before, norm_after, norm_ok ? "PASS" : "FAIL");
        if (!norm_ok) { ok = 0; }
    }

    /* positions 1,2,3 must differ (rotation must produce distinct encodings) */
    int distinct_ok = 1;
    for (int t1 = 0; t1 < T; t1++) {
        for (int t2 = t1+1; t2 < T; t2++) {
            float diff = 0.0f;
            for (int d = 0; d < Dh; d++) {
                float delta = Q_after[t1*Dh + d] - Q_after[t2*Dh + d];
                diff += delta * delta;
            }
            if (diff < 1e-8f) { distinct_ok = 0; } /* positions must produce distinct rotations */
        }
    }
    printf("  All positions produce distinct Q rotations: %s\n",
           distinct_ok ? "PASS" : "FAIL");
    if (!distinct_ok) { ok = 0; }

    printf("[05] RoPE Embedding %s\n", ok ? "PASS" : "FAIL");

    free(Q_before);
    free(Q_after);
    return 0;
}
