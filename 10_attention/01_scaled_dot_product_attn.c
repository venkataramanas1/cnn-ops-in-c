/* ============================================================================
 * [01] Scaled Dot-Product Attention (SDPA)
 * ----------------------------------------------------------------------------
 * The atomic kernel of ALL transformers.
 *
 *   Attn(Q,K,V) = softmax(Q @ K^T / sqrt(Dh)) @ V
 *
 * Tensor layout: 4D [N][H][T][Dh]
 *   N  = batch size
 *   H  = number of heads
 *   Tq = query sequence length  (can differ from Tkv in cross-attention)
 *   Tkv= key/value sequence length
 *   Dh = head dimension = D / H
 *
 * Flat index for [N][H][T][Dh]:
 *   n*H*T*Dh + h*T*Dh + t*Dh + d
 *
 * T is dynamic — sequence length varies per sample in VLA inference.
 *
 * Models: ALL transformers — ViT, BERT, GPT, LLaMA, RT-2, OpenVLA, π0.
 *
 * Reference: Vaswani et al. "Attention Is All You Need", NeurIPS 2017.
 *
 * Build: gcc 01_scaled_dot_product_attn.c -o 01_sdpa -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---- helpers ---- */

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

/* numerically-stable softmax over the last axis (cols) of a 2-D [rows x cols] array */
static void softmax_rows(float *m, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {       /* r: one query position */
        float maxv = m[r*cols];
        for (int c = 1; c < cols; c++) {   /* find row max for numerical stability */
            if (m[r*cols + c] > maxv) {
                maxv = m[r*cols + c];
            }
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] = expf(m[r*cols + c] - maxv); /* subtract max → no overflow */
            sum += m[r*cols + c];
        }
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] /= sum;          /* normalize → each row sums to 1.0 */
        }
    }
}

/* ============================================================
 * sdpa: Scaled Dot-Product Attention
 *   Q   [N][H][Tq][Dh]
 *   K   [N][H][Tkv][Dh]
 *   V   [N][H][Tkv][Dh]
 *   out [N][H][Tq][Dh]
 *
 * T is dynamic — strides are computed at call time.
 * ============================================================ */
static void sdpa(
    const float *Q, const float *K, const float *V, float *out,
    int N, int H, int Tq, int Tkv, int Dh)
{
    /* T is dynamic — sequence length varies per sample in VLA inference */
    int stride_Dh   = 1;                /* innermost: feature dimension */
    int stride_Tq   = Dh;              /* depends on runtime Dh */
    int stride_Tkv  = Dh;
    int stride_H_q  = Tq  * Dh;        /* depends on runtime Tq — computed at call time */
    int stride_H_kv = Tkv * Dh;        /* depends on runtime Tkv — computed at call time */
    int stride_N_q  = H * Tq  * Dh;
    int stride_N_kv = H * Tkv * Dh;

    float scale = 1.0f / sqrtf((float)Dh); /* 1/sqrt(Dh): controls score magnitude */

    /* scratch buffers for one (h, tq) slice */
    float *scores = malloc(Tq * Tkv * sizeof(float)); /* score matrix per head */

    for (int n = 0; n < N; n++) {              /* n: batch index */
        for (int h = 0; h < H; h++) {          /* h: head index */

            /* ---- Step 1: compute raw scores ----
             * scores[tq][tkv] = dot(Q[n,h,tq,:], K[n,h,tkv,:]) / sqrt(Dh)
             * dot product of query tq with key tkv; divide by sqrt(Dh) to control magnitude */
            for (int tq = 0; tq < Tq; tq++) {
                for (int tkv = 0; tkv < Tkv; tkv++) {
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; d++) {
                        /* Q flat index: n*stride_N_q + h*stride_H_q + tq*stride_Tq + d */
                        float qval = Q[n*stride_N_q  + h*stride_H_q  + tq*stride_Tq  + d];
                        /* K flat index: n*stride_N_kv + h*stride_H_kv + tkv*stride_Tkv + d */
                        float kval = K[n*stride_N_kv + h*stride_H_kv + tkv*stride_Tkv + d];
                        dot += qval * kval; /* accumulate inner product over Dh */
                    }
                    scores[tq*Tkv + tkv] = dot * scale; /* scale by 1/sqrt(Dh) */
                }
            }

            /* ---- Step 2: softmax over tkv axis per (h, tq) ----
             * after softmax, scores[h][tq] is a prob distribution over key positions */
            softmax_rows(scores, Tq, Tkv);

            /* ---- Step 3: weighted sum of values ----
             * out[n,h,tq,d] = sum_{tkv} scores[tq][tkv] * V[n,h,tkv,d]
             * weighted sum of values — attention score says how much each value slot contributes */
            for (int tq = 0; tq < Tq; tq++) {
                for (int d = 0; d < Dh; d++) {
                    float acc = 0.0f;
                    for (int tkv = 0; tkv < Tkv; tkv++) {
                        /* V flat index: n*stride_N_kv + h*stride_H_kv + tkv*stride_Tkv + d */
                        acc += scores[tq*Tkv + tkv]
                             * V[n*stride_N_kv + h*stride_H_kv + tkv*stride_Tkv + d];
                    }
                    /* out flat index: n*stride_N_q + h*stride_H_q + tq*stride_Tq + d */
                    out[n*stride_N_q + h*stride_H_q + tq*stride_Tq + d] = acc;
                }
            }
        }
    }
    free(scores);
}

/* ============================================================
 * Demo: N=1, H=2, Tq=3, Tkv=3, Dh=4
 * Q: ramp values, K: different ramp, V: ones + channel_id
 * ============================================================ */
int main(void)
{
    int N = 1, H = 2, Tq = 3, Tkv = 3, Dh = 4;
    int sz_Q = N * H * Tq  * Dh;
    int sz_KV= N * H * Tkv * Dh;

    float *Q   = calloc(sz_Q,  sizeof(float));
    float *K   = calloc(sz_KV, sizeof(float));
    float *V   = calloc(sz_KV, sizeof(float));
    float *out = calloc(sz_Q,  sizeof(float));

    /* Q: ramp 0.1, 0.2, ... across all elements */
    for (int i = 0; i < sz_Q; i++) {
        Q[i] = (i + 1) * 0.1f; /* query ramp — distinct from keys */
    }

    /* K: different ramp starting from 0.2 step 0.15 */
    for (int i = 0; i < sz_KV; i++) {
        K[i] = (i + 1) * 0.15f; /* key ramp — offset so Q·K is not trivially symmetric */
    }

    /* V: ones + channel_id so output is interpretable */
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int t = 0; t < Tkv; t++) {
                for (int d = 0; d < Dh; d++) {
                    /* V[n,h,t,d] = 1.0 + d: base 1 so weighted sum > 0; channel id makes channels distinguishable */
                    V[n*H*Tkv*Dh + h*Tkv*Dh + t*Dh + d] = 1.0f + (float)d;
                }
            }
        }
    }

    printf("=== Scaled Dot-Product Attention Demo ===\n");
    printf("N=%d H=%d Tq=%d Tkv=%d Dh=%d\n\n", N, H, Tq, Tkv, Dh);

    /* sqrt(Dh) scaling: without it, for large Dh the dot products grow large
     * → softmax saturates → vanishing gradients. Vaswani et al. attention paper, 2017. */
    printf("sqrt(Dh) scale factor = 1/sqrt(%d) = %.4f\n\n", Dh, 1.0f/sqrtf((float)Dh));

    /* show Q head 0 */
    print_grid("Q  head=0  [Tq x Dh]", Q, Tq, Dh);
    print_grid("K  head=0  [Tkv x Dh]", K, Tkv, Dh);
    print_grid("V  head=0  [Tkv x Dh]", V, Tkv, Dh);

    /* compute scores before softmax for display */
    float scale = 1.0f / sqrtf((float)Dh);
    float raw_scores[9]; /* Tq*Tkv = 9 */
    for (int tq = 0; tq < Tq; tq++) {
        for (int tkv = 0; tkv < Tkv; tkv++) {
            float dot = 0.0f;
            for (int d = 0; d < Dh; d++) {
                dot += Q[0*H*Tq*Dh + 0*Tq*Dh + tq*Dh + d]
                     * K[0*H*Tkv*Dh + 0*Tkv*Dh + tkv*Dh + d];
            }
            raw_scores[tq*Tkv + tkv] = dot * scale; /* raw score = Q·K / sqrt(Dh) */
        }
    }
    print_grid("Scores head=0 BEFORE softmax  [Tq x Tkv]", raw_scores, Tq, Tkv);

    /* run full SDPA */
    sdpa(Q, K, V, out, N, H, Tq, Tkv, Dh);

    /* show softmax scores (re-compute for display) */
    float post_scores[9];
    for (int i = 0; i < 9; i++) {
        post_scores[i] = raw_scores[i];
    }
    softmax_rows(post_scores, Tq, Tkv);
    print_grid("Scores head=0 AFTER softmax   [Tq x Tkv]", post_scores, Tq, Tkv);

    print_grid("Output head=0  [Tq x Dh]", out, Tq, Dh);
    print_grid("Output head=1  [Tq x Dh]", out + 1*Tq*Dh, Tq, Dh);

    /* ---- checks ---- */
    printf("--- Checks ---\n");
    /* softmax rows sum to 1 */
    int softmax_ok = 1;
    for (int tq = 0; tq < Tq; tq++) {
        float s = 0.0f;
        for (int tkv = 0; tkv < Tkv; tkv++) {
            s += post_scores[tq*Tkv + tkv];
        }
        printf("  head=0 tq=%d softmax row sum = %.6f  %s\n",
               tq, s, (fabsf(s - 1.0f) < 1e-5f) ? "PASS" : "FAIL");
        if (fabsf(s - 1.0f) > 1e-5f) {
            softmax_ok = 0;
        }
    }
    /* output row norms finite */
    int norms_ok = 1;
    for (int tq = 0; tq < Tq; tq++) {
        float norm = 0.0f;
        for (int d = 0; d < Dh; d++) {
            float v = out[tq*Dh + d];
            norm += v * v;
        }
        norm = sqrtf(norm);
        printf("  head=0 tq=%d output norm = %.4f  %s\n",
               tq, norm, (isfinite(norm) && norm > 0.0f) ? "PASS" : "FAIL");
        if (!isfinite(norm) || norm <= 0.0f) {
            norms_ok = 0;
        }
    }
    printf("[01] SDPA %s\n", (softmax_ok && norms_ok) ? "PASS" : "FAIL");

    free(Q); free(K); free(V); free(out);
    return 0;
}
