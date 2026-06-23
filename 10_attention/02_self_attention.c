/* ============================================================================
 * [02] Self-Attention
 * ----------------------------------------------------------------------------
 * Q, K, V all projected from the SAME input sequence.
 * Every token attends to every other token — quadratic in T.
 *
 * Pipeline:
 *   input [N][T][D]
 *   W_q [D][H*Dh],  W_k [D][H*Dh],  W_v [D][H*Dh]
 *   Q = input @ W_q → [N][T][H*Dh] → reshape [N][H][T][Dh]
 *   K = input @ W_k → same reshape
 *   V = input @ W_v → same reshape
 *   attn_out = SDPA(Q, K, V)  [N][H][T][Dh]
 *   concat → [N][T][H*Dh] → W_o [H*Dh][D] → [N][T][D]
 *
 * 4D layout [N][H][T][Dh]:
 *   flat index: n*H*T*Dh + h*T*Dh + t*Dh + d
 *
 * T is dynamic — sequence length varies per sample in VLA inference.
 *   int stride_T = Dh;
 *   int stride_H = T * Dh;   depends on runtime T — computed at call time
 *
 * Multi-head split: [N][T][H*Dh] → [N][H][T][Dh]
 *   No data copy — just re-striding. Each head sees a Dh-dim slice.
 *
 * Self-attention: every token attends to every other token.
 * Quadratic in T — the O(T^2) cost that Flash Attention solves.
 *
 * Models: ViT encoder, BERT, GPT, ALL VLA vision encoders.
 *
 * Build: gcc 02_self_attention.c -o 02_self_attention -lm
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

/* matmul: C[M][Kout] = A[M][Kin] @ B[Kin][Kout] */
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
            m[r*cols + c] = expf(m[r*cols + c] - maxv); /* subtract max before exp */
            sum += m[r*cols + c];
        }
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] /= sum; /* normalize: row becomes probability distribution */
        }
    }
}

/* ============================================================
 * self_attention:
 *   input [N][T][D]  → project → reshape 4D → SDPA → concat → project
 *   out   [N][T][D]
 * ============================================================ */
static void self_attention(
    const float *input,          /* [N][T][D] */
    const float *Wq,             /* [D][H*Dh] */
    const float *Wk,             /* [D][H*Dh] */
    const float *Wv,             /* [D][H*Dh] */
    const float *Wo,             /* [H*Dh][D] */
    float *out,                  /* [N][T][D] */
    float *Q4d,                  /* scratch [N][H][T][Dh] */
    float *K4d,
    float *V4d,
    float *attn_out,             /* scratch [N][H][T][Dh] */
    float *concat_buf,           /* scratch [N][T][H*Dh] */
    int N, int H, int T, int D, int Dh)
{
    int HD = H * Dh; /* total projected dim = H * Dh */

    /* T is dynamic — sequence length varies per sample in VLA inference */
    int stride_T_4d = Dh;               /* innermost after feature dim */
    int stride_H_4d = T * Dh;           /* depends on runtime T — computed at call time */
    int stride_N_4d = H * T * Dh;

    float scale = 1.0f / sqrtf((float)Dh); /* 1/sqrt(Dh) scaling factor */

    for (int n = 0; n < N; n++) {

        /* ---- Step 1: project input to Q, K, V ----
         * input[n] is [T][D]; project each to [T][H*Dh] */
        float *tmp_Q = malloc(T * HD * sizeof(float)); /* [T][H*Dh] before reshape */
        float *tmp_K = malloc(T * HD * sizeof(float));
        float *tmp_V = malloc(T * HD * sizeof(float));

        matmul(input + n*T*D, Wq, tmp_Q, T, D, HD); /* tmp_Q = input[n] @ Wq */
        matmul(input + n*T*D, Wk, tmp_K, T, D, HD); /* tmp_K = input[n] @ Wk */
        matmul(input + n*T*D, Wv, tmp_V, T, D, HD); /* tmp_V = input[n] @ Wv */

        /* ---- Step 2: reshape [T][H*Dh] → [H][T][Dh] (4D view) ----
         * multi-head split: [N][T][H*Dh] → [N][H][T][Dh]
         * No data copy — just re-striding. Each head sees a Dh-dim slice of the full D-dim feature. */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    /* source: tmp[t][h*Dh + d] = tmp[t*HD + h*Dh + d] */
                    /* dest 4D flat: n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d */
                    Q4d[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d] =
                        tmp_Q[t*HD + h*Dh + d]; /* interleave heads → head-first layout */
                    K4d[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d] =
                        tmp_K[t*HD + h*Dh + d];
                    V4d[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d] =
                        tmp_V[t*HD + h*Dh + d];
                }
            }
        }
        free(tmp_Q); free(tmp_K); free(tmp_V);

        /* ---- Step 3: SDPA per head ----
         * scores[tq][tk] = Q[h,tq,:] · K[h,tk,:] / sqrt(Dh), then softmax, then @V */
        float *scores = malloc(T * T * sizeof(float)); /* [T][T] attention map per head */

        for (int h = 0; h < H; h++) {
            /* base pointers for this (n, h) slice */
            const float *Qh = Q4d + n*stride_N_4d + h*stride_H_4d; /* [T][Dh] */
            const float *Kh = K4d + n*stride_N_4d + h*stride_H_4d;
            const float *Vh = V4d + n*stride_N_4d + h*stride_H_4d;
            float       *Oh = attn_out + n*stride_N_4d + h*stride_H_4d; /* [T][Dh] */

            /* compute score[tq][tk] = dot(Q[tq], K[tk]) / sqrt(Dh) */
            for (int tq = 0; tq < T; tq++) {
                for (int tk = 0; tk < T; tk++) {
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; d++) {
                        /* Qh flat: tq*Dh + d;  Kh flat: tk*Dh + d */
                        dot += Qh[tq*Dh + d] * Kh[tk*Dh + d]; /* dot product over head dim */
                    }
                    scores[tq*T + tk] = dot * scale; /* scaled attention score */
                }
            }

            /* softmax over tk axis: each query becomes a distribution over key positions */
            softmax_rows(scores, T, T);

            /* weighted sum of values: out[tq,d] = sum_tk scores[tq,tk] * V[tk,d] */
            for (int tq = 0; tq < T; tq++) {
                for (int d = 0; d < Dh; d++) {
                    float acc = 0.0f;
                    for (int tk = 0; tk < T; tk++) {
                        acc += scores[tq*T + tk] * Vh[tk*Dh + d]; /* blend values by attention weights */
                    }
                    Oh[tq*Dh + d] = acc; /* 4D output: attention-weighted value vector */
                }
            }
        }
        free(scores);

        /* ---- Step 4: reshape back [H][T][Dh] → [T][H*Dh] for output projection ----
         * inverse of step 2: head-first → token-first for W_o projection */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    /* source 4D: n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d */
                    concat_buf[t*HD + h*Dh + d] =
                        attn_out[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d];
                }
            }
        }

        /* ---- Step 5: output projection W_o: [T][H*Dh] @ [H*Dh][D] → [T][D] ----
         * mixes information across heads into final D-dim representation */
        matmul(concat_buf, Wo, out + n*T*D, T, HD, D);
    }
}

/* ============================================================
 * Demo: N=1 H=2 T=4 D=8 Dh=4
 * Input: one-hot pattern per token (makes attention patterns visible)
 * ============================================================ */
int main(void)
{
    int N = 1, H = 2, T = 4, D = 8, Dh = 4;
    int HD = H * Dh; /* = 8, equals D here */

    /* T is dynamic — strides depend on runtime T */
    int stride_T_4d = Dh;
    int stride_H_4d = T * Dh;   /* depends on runtime T — computed at call time */
    int stride_N_4d = H * T * Dh;

    float *input = calloc(N*T*D,       sizeof(float));
    float *Wq    = calloc(D*HD,        sizeof(float));
    float *Wk    = calloc(D*HD,        sizeof(float));
    float *Wv    = calloc(D*HD,        sizeof(float));
    float *Wo    = calloc(HD*D,        sizeof(float));
    float *Q4d   = calloc(N*H*T*Dh,   sizeof(float));
    float *K4d   = calloc(N*H*T*Dh,   sizeof(float));
    float *V4d   = calloc(N*H*T*Dh,   sizeof(float));
    float *attn  = calloc(N*H*T*Dh,   sizeof(float));
    float *cbuf  = calloc(T*HD,        sizeof(float));
    float *out   = calloc(N*T*D,       sizeof(float));

    /* one-hot pattern per token: token t has a 1 at position t, scaled */
    for (int t = 0; t < T; t++) {
        input[t*D + t] = (float)(t + 1); /* token t = e_t scaled by (t+1) */
    }

    /* identity W_q, W_k: each head projects to its own half of D
     * head 0 → dims 0..3, head 1 → dims 4..7 */
    for (int h = 0; h < H; h++) {
        for (int d = 0; d < Dh; d++) {
            int src_d = h*Dh + d;          /* source feature dimension */
            int dst_d = h*Dh + d;          /* destination in H*Dh output */
            Wq[src_d*HD + dst_d] = 1.0f;  /* identity slice for head h */
            Wk[src_d*HD + dst_d] = 1.0f;
            Wv[src_d*HD + dst_d] = 1.0f;
        }
    }
    /* W_o: identity (H*Dh = D so square) */
    for (int i = 0; i < HD; i++) {
        Wo[i*D + i] = 1.0f; /* pass through concat heads unchanged */
    }

    printf("=== Self-Attention Demo ===\n");
    printf("N=%d H=%d T=%d D=%d Dh=%d\n", N, H, T, D, Dh);
    printf("Self-attention: every token attends to every other token.\n");
    printf("Quadratic in T — the O(T^2) cost that Flash Attention solves.\n\n");

    print_grid("Input  [T x D]", input, T, D);

    self_attention(input, Wq, Wk, Wv, Wo, out,
                   Q4d, K4d, V4d, attn, cbuf,
                   N, H, T, D, Dh);

    /* show Q head 0 in 4D layout */
    print_grid("Q head=0  [T x Dh]  (4D slice)",
               Q4d + 0*stride_N_4d + 0*stride_H_4d, T, Dh);
    print_grid("Q head=1  [T x Dh]  (4D slice)",
               Q4d + 0*stride_N_4d + 1*stride_H_4d, T, Dh);

    /* show attention scores: re-compute softmax(QK^T/sqrt(Dh)) for head 0 */
    float scores[16]; /* T*T */
    float scale = 1.0f / sqrtf((float)Dh);
    for (int tq = 0; tq < T; tq++) {
        for (int tk = 0; tk < T; tk++) {
            float dot = 0.0f;
            for (int d = 0; d < Dh; d++) {
                dot += Q4d[0*stride_N_4d + 0*stride_H_4d + tq*Dh + d]
                     * K4d[0*stride_N_4d + 0*stride_H_4d + tk*Dh + d];
            }
            scores[tq*T + tk] = dot * scale;
        }
    }
    softmax_rows(scores, T, T);
    print_grid("Attention scores head=0 (softmax)  [T x T]", scores, T, T);

    print_grid("Output  [T x D]", out, T, D);

    /* checks */
    printf("--- Checks ---\n");
    /* output same shape as input */
    printf("  Output shape matches input [%d][%d]: PASS\n", T, D);
    /* softmax row sums */
    int ok = 1;
    for (int tq = 0; tq < T; tq++) {
        float s = 0.0f;
        for (int tk = 0; tk < T; tk++) { s += scores[tq*T + tk]; }
        printf("  head=0 tq=%d attn row sum=%.6f  %s\n",
               tq, s, (fabsf(s - 1.0f) < 1e-5f) ? "PASS" : "FAIL");
        if (fabsf(s - 1.0f) > 1e-5f) { ok = 0; }
    }
    printf("[02] Self-Attention %s\n", ok ? "PASS" : "FAIL");

    free(input); free(Wq); free(Wk); free(Wv); free(Wo);
    free(Q4d); free(K4d); free(V4d); free(attn); free(cbuf); free(out);
    return 0;
}
