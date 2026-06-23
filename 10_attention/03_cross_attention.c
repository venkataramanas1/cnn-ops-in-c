/* ============================================================================
 * [03] Cross-Attention
 * ----------------------------------------------------------------------------
 * Q from one sequence, K and V from a DIFFERENT sequence.
 *
 *   query_seq  [N][Tq][D]  — e.g. robot action tokens / decoder tokens
 *   context_seq[N][Tkv][D] — e.g. vision tokens or language tokens
 *
 *   Q = query_seq   @ W_q  → [N][Tq][H*Dh]  → reshape [N][H][Tq][Dh]
 *   K = context_seq @ W_k  → [N][Tkv][H*Dh] → reshape [N][H][Tkv][Dh]
 *   V = context_seq @ W_v  → same
 *   out = SDPA(Q, K, V)    → [N][H][Tq][Dh] → reshape → [N][Tq][D]
 *
 * 4D layout [N][H][T][Dh]:
 *   flat index: n*H*T*Dh + h*T*Dh + t*Dh + d
 *
 * T is dynamic — sequence length varies per sample in VLA inference.
 *   int stride_T = Dh;
 *   int stride_H = T * Dh;   depends on runtime T — computed at call time
 *
 * Cross-attention is HOW VLA models fuse modalities:
 *   the robot action query "asks" the vision-language context "what should I do here?"
 *   The context provides Keys and Values; the action query provides the Query.
 *
 * Models:
 *   RT-2      — decoder cross-attends to image tokens
 *   OpenVLA   — cross-modal fusion
 *   π0        — diffusion policy cross-attends to language
 *   Flamingo  — image-text cross-attention
 *
 * Build: gcc 03_cross_attention.c -o 03_cross_attention -lm
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

static void matmul(const float *A, const float *B, float *C,
                   int M, int Kin, int Kout)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Kout; j++) {
            float s = 0.0f;
            for (int k = 0; k < Kin; k++) {
                s += A[i*Kin + k] * B[k*Kout + j]; /* A[i][k] * B[k][j] */
            }
            C[i*Kout + j] = s;
        }
    }
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
            m[r*cols + c] /= sum; /* normalize to probability distribution */
        }
    }
}

/* ============================================================
 * cross_attention:
 *   query_seq   [N][Tq][D]
 *   context_seq [N][Tkv][D]
 *   W_q [D][H*Dh], W_k [D][H*Dh], W_v [D][H*Dh], W_o [H*Dh][D]
 *   out [N][Tq][D]
 *   attn_map [Tq][Tkv] — attention map for head 0 of sample 0 (for display)
 * ============================================================ */
static void cross_attention(
    const float *query_seq,   /* [N][Tq][D] */
    const float *context_seq, /* [N][Tkv][D] */
    const float *Wq,          /* [D][H*Dh] */
    const float *Wk,          /* [D][H*Dh] */
    const float *Wv,          /* [D][H*Dh] */
    const float *Wo,          /* [H*Dh][D] */
    float *out,               /* [N][Tq][D] */
    float *attn_map,          /* [Tq][Tkv] for display */
    int N, int H, int Tq, int Tkv, int D, int Dh)
{
    int HD = H * Dh; /* concat head dimension */

    /* T is dynamic — sequence length varies per sample in VLA inference */
    int stride_Tq_4d  = Dh;               /* feature-level stride */
    int stride_Tkv_4d = Dh;
    int stride_H_q    = Tq  * Dh;         /* depends on runtime Tq — computed at call time */
    int stride_H_kv   = Tkv * Dh;         /* depends on runtime Tkv — computed at call time */
    int stride_N_q    = H * Tq  * Dh;
    int stride_N_kv   = H * Tkv * Dh;

    float scale = 1.0f / sqrtf((float)Dh); /* 1/sqrt(Dh) scaling */

    float *Q4d   = calloc(N * H * Tq  * Dh, sizeof(float)); /* [N][H][Tq][Dh] */
    float *K4d   = calloc(N * H * Tkv * Dh, sizeof(float)); /* [N][H][Tkv][Dh] */
    float *V4d   = calloc(N * H * Tkv * Dh, sizeof(float));
    float *aout  = calloc(N * H * Tq  * Dh, sizeof(float)); /* [N][H][Tq][Dh] */
    float *cbuf  = calloc(Tq * HD,           sizeof(float)); /* [Tq][H*Dh] */
    float *scores= calloc(Tq * Tkv,          sizeof(float)); /* [Tq][Tkv] per head */
    float *tmpQ  = malloc(Tq  * HD * sizeof(float)); /* projected before reshape */
    float *tmpKV = malloc(Tkv * HD * sizeof(float));

    for (int n = 0; n < N; n++) {

        /* ---- Step 1: project queries and context ----
         * Q comes from query_seq (action tokens); K,V from context_seq (vision/language) */
        matmul(query_seq   + n*Tq*D,  Wq, tmpQ,  Tq,  D, HD); /* Q = query   @ W_q */
        matmul(context_seq + n*Tkv*D, Wk, tmpKV, Tkv, D, HD); /* K = context @ W_k */
        /* reshape [Tq][H*Dh] → [H][Tq][Dh] into Q4d */
        for (int t = 0; t < Tq; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    Q4d[n*stride_N_q + h*stride_H_q + t*stride_Tq_4d + d] =
                        tmpQ[t*HD + h*Dh + d]; /* token-first → head-first reshape */
                }
            }
        }
        /* reshape [Tkv][H*Dh] → [H][Tkv][Dh] into K4d */
        for (int t = 0; t < Tkv; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    K4d[n*stride_N_kv + h*stride_H_kv + t*stride_Tkv_4d + d] =
                        tmpKV[t*HD + h*Dh + d];
                }
            }
        }
        /* project V from context */
        matmul(context_seq + n*Tkv*D, Wv, tmpKV, Tkv, D, HD); /* V = context @ W_v */
        for (int t = 0; t < Tkv; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    V4d[n*stride_N_kv + h*stride_H_kv + t*stride_Tkv_4d + d] =
                        tmpKV[t*HD + h*Dh + d];
                }
            }
        }

        /* ---- Step 2: SDPA — Q from actions, K/V from context ----
         * This is the cross-attention operation: action query "queries" the vision/language context */
        for (int h = 0; h < H; h++) {
            const float *Qh = Q4d + n*stride_N_q  + h*stride_H_q;  /* [Tq][Dh] */
            const float *Kh = K4d + n*stride_N_kv + h*stride_H_kv; /* [Tkv][Dh] */
            const float *Vh = V4d + n*stride_N_kv + h*stride_H_kv;
            float       *Oh = aout + n*stride_N_q + h*stride_H_q;   /* [Tq][Dh] */

            /* scores[tq][tkv] = Q[tq] · K[tkv] / sqrt(Dh)
             * which context tokens each action query should attend to */
            for (int tq = 0; tq < Tq; tq++) {
                for (int tkv = 0; tkv < Tkv; tkv++) {
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; d++) {
                        dot += Qh[tq*Dh + d] * Kh[tkv*Dh + d]; /* inner product over head dim */
                    }
                    scores[tq*Tkv + tkv] = dot * scale; /* scaled score */
                }
            }

            /* softmax over context positions: action query becomes distribution over context */
            softmax_rows(scores, Tq, Tkv);

            /* save attention map for head 0 of sample 0 for display */
            if (n == 0 && h == 0) {
                for (int i = 0; i < Tq * Tkv; i++) {
                    attn_map[i] = scores[i]; /* copy attention map for visualization */
                }
            }

            /* weighted sum: out[tq] = sum_tkv attn[tq,tkv] * V[tkv] */
            for (int tq = 0; tq < Tq; tq++) {
                for (int d = 0; d < Dh; d++) {
                    float acc = 0.0f;
                    for (int tkv = 0; tkv < Tkv; tkv++) {
                        acc += scores[tq*Tkv + tkv] * Vh[tkv*Dh + d]; /* blend context values */
                    }
                    Oh[tq*Dh + d] = acc; /* action token updated with context information */
                }
            }
        }

        /* ---- Step 3: reshape [H][Tq][Dh] → [Tq][H*Dh] → W_o → [Tq][D] ---- */
        for (int t = 0; t < Tq; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    cbuf[t*HD + h*Dh + d] =
                        aout[n*stride_N_q + h*stride_H_q + t*stride_Tq_4d + d];
                }
            }
        }
        matmul(cbuf, Wo, out + n*Tq*D, Tq, HD, D); /* output projection mixes heads */
    }

    free(Q4d); free(K4d); free(V4d); free(aout); free(cbuf);
    free(scores); free(tmpQ); free(tmpKV);
}

/* ============================================================
 * Demo: N=1, Tq=2, Tkv=4, H=1, D=4, Dh=4
 * query_seq:   2 action query tokens (learnable — demo as specific values)
 * context_seq: 4 visual/language context tokens
 * ============================================================ */
int main(void)
{
    int N = 1, H = 1, Tq = 2, Tkv = 4, D = 4, Dh = 4;
    int HD = H * Dh;

    float *query_seq   = calloc(N*Tq*D,  sizeof(float));
    float *context_seq = calloc(N*Tkv*D, sizeof(float));
    float *Wq          = calloc(D*HD,    sizeof(float));
    float *Wk          = calloc(D*HD,    sizeof(float));
    float *Wv          = calloc(D*HD,    sizeof(float));
    float *Wo          = calloc(HD*D,    sizeof(float));
    float *out         = calloc(N*Tq*D,  sizeof(float));
    float *attn_map    = calloc(Tq*Tkv,  sizeof(float));

    /* action query tokens: learnable vectors (demo as specific non-trivial values) */
    /* query 0: attends to "action start" context — high on dims 0,1 */
    query_seq[0*D + 0] = 1.0f;
    query_seq[0*D + 1] = 0.5f;
    query_seq[0*D + 2] = 0.0f;
    query_seq[0*D + 3] = 0.0f;
    /* query 1: attends to "gripper" context — high on dims 2,3 */
    query_seq[1*D + 0] = 0.0f;
    query_seq[1*D + 1] = 0.0f;
    query_seq[1*D + 2] = 1.0f;
    query_seq[1*D + 3] = 0.8f;

    /* context tokens: 4 vision/language tokens with distinct features */
    /* token 0: "object location" feature — strong on dim 0 */
    context_seq[0*D + 0] = 2.0f;  context_seq[0*D + 1] = 1.0f;
    /* token 1: "action verb" feature — strong on dim 1 */
    context_seq[1*D + 0] = 0.5f;  context_seq[1*D + 1] = 2.0f;
    /* token 2: "gripper pose" feature — strong on dims 2,3 */
    context_seq[2*D + 2] = 2.0f;  context_seq[2*D + 3] = 1.5f;
    /* token 3: "background" — weak uniform signal */
    context_seq[3*D + 0] = 0.1f;  context_seq[3*D + 1] = 0.1f;
    context_seq[3*D + 2] = 0.1f;  context_seq[3*D + 3] = 0.1f;

    /* identity projections (D = H*Dh = 4) */
    for (int i = 0; i < D; i++) {
        Wq[i*HD + i] = 1.0f; /* pass through: Q = query_seq unchanged */
        Wk[i*HD + i] = 1.0f; /* pass through: K = context_seq unchanged */
        Wv[i*HD + i] = 1.0f; /* pass through: V = context_seq unchanged */
        Wo[i*D  + i] = 1.0f; /* pass through output projection */
    }

    printf("=== Cross-Attention Demo ===\n");
    printf("N=%d H=%d Tq=%d Tkv=%d D=%d Dh=%d\n", N, H, Tq, Tkv, D, Dh);
    printf("Cross-attention: Q from action tokens, K/V from vision/language context.\n");
    printf("VLA fusion: robot action query asks the context 'what should I do here?'\n\n");

    print_grid("Query tokens (action)  [Tq x D]", query_seq, Tq, D);
    print_grid("Context tokens (vision/lang)  [Tkv x D]", context_seq, Tkv, D);

    cross_attention(query_seq, context_seq, Wq, Wk, Wv, Wo,
                    out, attn_map, N, H, Tq, Tkv, D, Dh);

    /* show attention map [Tq][Tkv] — which context tokens each action query attends to */
    print_grid("Attention map  [Tq x Tkv]  (which context each action attends to)",
               attn_map, Tq, Tkv);

    print_grid("Output (action tokens updated with context)  [Tq x D]", out, Tq, D);

    /* checks: attention rows sum to 1 */
    printf("--- Checks ---\n");
    int ok = 1;
    for (int tq = 0; tq < Tq; tq++) {
        float s = 0.0f;
        for (int tkv = 0; tkv < Tkv; tkv++) {
            s += attn_map[tq*Tkv + tkv]; /* sum attention weights over context positions */
        }
        printf("  action query %d attn row sum = %.6f  %s\n",
               tq, s, (fabsf(s - 1.0f) < 1e-5f) ? "PASS" : "FAIL");
        if (fabsf(s - 1.0f) > 1e-5f) { ok = 0; }
    }
    printf("[03] Cross-Attention %s\n", ok ? "PASS" : "FAIL");

    free(query_seq); free(context_seq); free(Wq); free(Wk); free(Wv);
    free(Wo); free(out); free(attn_map);
    return 0;
}
