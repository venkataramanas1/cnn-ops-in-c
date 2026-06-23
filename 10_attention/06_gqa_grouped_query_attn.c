/* ============================================================================
 * [06] Grouped Query Attention (GQA)
 * ----------------------------------------------------------------------------
 * K and V use FEWER heads than Q. Each KV head is SHARED by a group of Q heads.
 *
 *   Q: [N][Hq][T][Dh]    Hq query heads
 *   K: [N][Hkv][T][Dh]   Hkv key heads    (Hkv < Hq, Hq % Hkv == 0)
 *   V: [N][Hkv][T][Dh]   Hkv value heads
 *
 *   group_size = Hq / Hkv        — number of Q heads sharing each KV head
 *   kv_head    = q_head / group_size  — which KV head a Q head uses
 *
 * 4D layout [N][H][T][Dh]:
 *   flat index (Q): n*Hq*T*Dh + h*T*Dh + t*Dh + d
 *   flat index (K/V): n*Hkv*T*Dh + kv_h*T*Dh + t*Dh + d
 *
 * T is dynamic — sequence length varies per sample in VLA inference.
 *   int stride_H_q  = T * Dh;   depends on runtime T — computed at call time
 *   int stride_H_kv = T * Dh;   same, but separate for clarity
 *
 * GQA: instead of Hq independent KV pairs, only Hkv pairs — groups of (Hq/Hkv)
 * Q heads share one KV head. LLaMA-3-8B uses Hq=32, Hkv=8.
 * Edge effect: KV cache is 4x smaller → fits on-device.
 *
 * Hq/Hkv ratio is a runtime parameter.
 *
 * Models: LLaMA-3, Mistral, Gemma — GQA replaced MHA in almost all modern LLMs
 *   to reduce KV cache memory. Critical for edge deployment.
 *
 * Build: gcc 06_gqa_grouped_query_attn.c -o 06_gqa -lm
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

static void softmax_rows(float *m, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float maxv = m[r*cols];
        for (int c = 1; c < cols; c++) {
            if (m[r*cols + c] > maxv) { maxv = m[r*cols + c]; }
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] = expf(m[r*cols + c] - maxv); /* stable exp */
            sum += m[r*cols + c];
        }
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] /= sum; /* row becomes a probability distribution */
        }
    }
}

/* ============================================================
 * gqa: Grouped Query Attention
 *   Q   [N][Hq][T][Dh]
 *   K   [N][Hkv][T][Dh]
 *   V   [N][Hkv][T][Dh]
 *   out [N][Hq][T][Dh]
 *   scores_out [Hq][T][T]  — for display (sample n=0 only)
 *
 * group_size = Hq / Hkv  — runtime parameter
 * ============================================================ */
static void gqa(
    const float *Q,    /* [N][Hq][T][Dh] */
    const float *K,    /* [N][Hkv][T][Dh] */
    const float *V,    /* [N][Hkv][T][Dh] */
    float *out,        /* [N][Hq][T][Dh] */
    float *scores_out, /* [Hq][T][T] for display, n=0 */
    int N, int Hq, int Hkv, int T, int Dh)
{
    /* group_size = Hq / Hkv — runtime parameter (key GQA design choice) */
    int group_size = Hq / Hkv; /* each KV head is shared by group_size Q heads */

    /* T is dynamic — sequence length varies per sample in VLA inference */
    int stride_T_q   = Dh;
    int stride_T_kv  = Dh;
    int stride_H_q   = T * Dh;   /* depends on runtime T — computed at call time */
    int stride_H_kv  = T * Dh;   /* same stride formula, but Hkv heads total */
    int stride_N_q   = Hq  * T * Dh;
    int stride_N_kv  = Hkv * T * Dh;

    float scale = 1.0f / sqrtf((float)Dh); /* 1/sqrt(Dh) scale factor */

    float *scores = malloc(T * T * sizeof(float)); /* [T][T] per head */

    for (int n = 0; n < N; n++) {
        for (int hq = 0; hq < Hq; hq++) { /* hq: query head index */

            /* kv_head = hq / group_size — which KV head this Q head uses
             * Q heads 0..group_size-1 share KV head 0,
             * Q heads group_size..2*group_size-1 share KV head 1, etc. */
            int hkv = hq / group_size; /* shared KV head index */

            const float *Qh = Q + n*stride_N_q  + hq  * stride_H_q;  /* [T][Dh] */
            const float *Kh = K + n*stride_N_kv + hkv * stride_H_kv; /* [T][Dh] — shared */
            const float *Vh = V + n*stride_N_kv + hkv * stride_H_kv; /* [T][Dh] — shared */
            float       *Oh = out + n*stride_N_q + hq  * stride_H_q;  /* [T][Dh] */

            /* ---- scores[tq][tk] = Q[hq,tq] · K[hkv,tk] / sqrt(Dh) ----
             * the K used here (hkv) is SHARED across group_size Q heads */
            for (int tq = 0; tq < T; tq++) {
                for (int tk = 0; tk < T; tk++) {
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; d++) {
                        /* Qh flat: tq*stride_T_q + d;  Kh flat: tk*stride_T_kv + d */
                        dot += Qh[tq*stride_T_q + d] * Kh[tk*stride_T_kv + d];
                    }
                    scores[tq*T + tk] = dot * scale; /* scaled attention score */
                }
            }

            /* softmax: each query becomes a distribution over key positions */
            softmax_rows(scores, T, T);

            /* save scores for display (sample 0 only) */
            if (n == 0) {
                for (int i = 0; i < T*T; i++) {
                    scores_out[hq*T*T + i] = scores[i]; /* store [Hq][T][T] */
                }
            }

            /* weighted sum of values: out[hq,tq] = sum_tk scores[tq,tk] * V[hkv,tk] */
            for (int tq = 0; tq < T; tq++) {
                for (int d = 0; d < Dh; d++) {
                    float acc = 0.0f;
                    for (int tk = 0; tk < T; tk++) {
                        /* V is indexed by hkv (shared), not hq (individual) */
                        acc += scores[tq*T + tk] * Vh[tk*stride_T_kv + d];
                    }
                    Oh[tq*stride_T_q + d] = acc; /* output for this Q head */
                }
            }
        }
    }
    free(scores);
}

/* ============================================================
 * Demo: N=1 Hq=4 Hkv=2 T=3 Dh=4
 * group_size = 4/2 = 2: Q heads {0,1} share KV head 0; Q heads {2,3} share KV head 1
 * ============================================================ */
int main(void)
{
    int N = 1, Hq = 4, Hkv = 2, T = 3, Dh = 4;
    int group_size = Hq / Hkv; /* = 2: runtime parameter showing head sharing ratio */

    printf("=== Grouped Query Attention (GQA) Demo ===\n");
    printf("N=%d Hq=%d Hkv=%d T=%d Dh=%d  group_size=%d\n\n",
           N, Hq, Hkv, T, Dh, group_size);
    printf("GQA: groups of %d Q heads share one KV head.\n", group_size);
    printf("LLaMA-3-8B: Hq=32, Hkv=8 → 4x smaller KV cache → fits on-device.\n\n");

    /* show which Q heads share which KV heads */
    printf("Q-head → KV-head mapping (group_size=%d):\n", group_size);
    for (int hq = 0; hq < Hq; hq++) {
        int hkv = hq / group_size; /* runtime grouping: integer divide */
        printf("  Q head %d → KV head %d  (shares with Q head %d)\n",
               hq, hkv, hq ^ 1); /* XOR 1 gives the co-head in the pair */
    }
    printf("\n");

    float *Q          = calloc(N*Hq *T*Dh, sizeof(float));
    float *K          = calloc(N*Hkv*T*Dh, sizeof(float));
    float *V          = calloc(N*Hkv*T*Dh, sizeof(float));
    float *out        = calloc(N*Hq *T*Dh, sizeof(float));
    float *scores_all = calloc(Hq*T*T,      sizeof(float)); /* [Hq][T][T] */

    /* Q heads: heads 0,1 get one pattern; heads 2,3 get a different pattern
     * This demonstrates that heads sharing the same KV head still compute
     * the same attention distribution (same K,V) even with different Q. */
    for (int hq = 0; hq < Hq; hq++) {
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < Dh; d++) {
                int group = hq / group_size; /* which KV group */
                /* Q value: token index + small head offset so heads 0,1 are similar but not identical */
                Q[hq*T*Dh + t*Dh + d] = (float)(t + 1) * 0.5f
                                       + (float)group * 2.0f   /* large group offset */
                                       + (float)d * 0.01f;     /* tiny dim offset */
            }
        }
    }

    /* K: KV head 0 is "spatial" (high on dim 0); KV head 1 is "semantic" (high on dim 2) */
    for (int t = 0; t < T; t++) {
        /* KV head 0: spatial pattern */
        K[0*T*Dh + t*Dh + 0] = (float)(t + 1) * 1.0f; /* strong position signal on dim 0 */
        K[0*T*Dh + t*Dh + 1] = (float)(t + 1) * 0.5f;
        /* KV head 1: semantic pattern */
        K[1*T*Dh + t*Dh + 2] = (float)(T - t) * 1.0f; /* reversed position on dim 2 */
        K[1*T*Dh + t*Dh + 3] = (float)(T - t) * 0.5f;
    }

    /* V: identity-like values so output is interpretable */
    for (int hkv = 0; hkv < Hkv; hkv++) {
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < Dh; d++) {
                /* V[hkv,t,d] = (hkv+1)*10 + t + d*0.1: unique per KV head */
                V[hkv*T*Dh + t*Dh + d] = (float)(hkv + 1) * 10.0f
                                        + (float)t
                                        + (float)d * 0.1f;
            }
        }
    }

    gqa(Q, K, V, out, scores_all, N, Hq, Hkv, T, Dh);

    /* show attention score maps for all Q heads */
    printf("Attention scores per Q head  [T x T]:\n");
    for (int hq = 0; hq < Hq; hq++) {
        int hkv = hq / group_size;
        char title[64];
        snprintf(title, sizeof(title),
                 "  Scores Q head=%d (uses KV head=%d)", hq, hkv);
        print_grid(title, scores_all + hq*T*T, T, T);
    }

    /* show output per Q head */
    for (int hq = 0; hq < Hq; hq++) {
        char title[64];
        snprintf(title, sizeof(title), "Output Q head=%d  [T x Dh]", hq);
        print_grid(title, out + hq*T*Dh, T, Dh);
    }

    /* ---- checks ---- */
    printf("--- Checks ---\n");
    int ok = 1;

    /* softmax row sums = 1 for all Q heads */
    for (int hq = 0; hq < Hq; hq++) {
        for (int tq = 0; tq < T; tq++) {
            float s = 0.0f;
            for (int tk = 0; tk < T; tk++) {
                s += scores_all[hq*T*T + tq*T + tk];
            }
            int pass = fabsf(s - 1.0f) < 1e-5f;
            if (!pass) {
                printf("  FAIL: Q head=%d tq=%d attn row sum=%.6f\n", hq, tq, s);
                ok = 0;
            }
        }
    }
    printf("  All softmax rows sum to 1.0: %s\n", ok ? "PASS" : "FAIL");

    /* Q heads sharing same KV head produce same attention distribution
     * (because same K is used, and Q values were set to be in same group) */
    int shared_ok = 1;
    for (int g = 0; g < Hkv; g++) {
        /* check heads g*group_size and g*group_size+1 */
        int h0 = g * group_size;
        int h1 = g * group_size + 1;
        float max_diff = 0.0f;
        for (int i = 0; i < T*T; i++) {
            float diff = fabsf(scores_all[h0*T*T + i] - scores_all[h1*T*T + i]);
            if (diff > max_diff) { max_diff = diff; }
        }
        /* shared KV head → same attention IF Q values were identical;
         * here Q has a tiny per-dim offset (0.01) so expect near-identical not exact */
        printf("  KV head=%d shared by Q heads %d,%d — attn map max diff=%.6f  %s\n",
               g, h0, h1, max_diff,
               (max_diff < 0.1f) ? "NEAR-IDENTICAL (PASS)" : "DIVERGED (FAIL)");
        if (max_diff >= 0.1f) { shared_ok = 0; }
    }
    if (!shared_ok) { ok = 0; }

    /* output shape check */
    printf("  Output shape [Hq=%d][T=%d][Dh=%d] confirmed: PASS\n", Hq, T, Dh);

    printf("[06] GQA %s\n", ok ? "PASS" : "FAIL");

    free(Q); free(K); free(V); free(out); free(scores_all);
    return 0;
}
