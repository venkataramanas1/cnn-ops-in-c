/* ============================================================================
 * [04] Full MHSA Block = Self-Attention + LayerNorm + Residual
 * ----------------------------------------------------------------------------
 * The complete transformer encoder building block.
 *
 * Pre-norm variant (modern default — used in ViT-L / OpenVLA backbone):
 *   out = x + SelfAttention(LayerNorm(x))
 *
 * Post-norm variant (original Vaswani 2017):
 *   out = LayerNorm(x + SelfAttention(x))
 *
 * Both are shown. Demo uses pre-norm.
 *
 * 4D layout inside attention: [N][H][T][Dh]
 *   flat index: n*H*T*Dh + h*T*Dh + t*Dh + d
 *
 * T is dynamic — sequence length varies per sample in VLA inference.
 *   int stride_H = T * Dh;   depends on runtime T — computed at call time
 *
 * MHSA block = attention + residual + norm.
 *   Residual (add x back): ensures gradients flow even if attention learns identity.
 *   Pre-norm (norm before attention): more stable for deep models — used in ViT-L (OpenVLA backbone).
 *
 * Models: ViT block, DeiT, Swin Transformer, MobileViT.
 *
 * Build: gcc 04_mhsa_full.c -o 04_mhsa_full -lm
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
                s += A[i*Kin + k] * B[k*Kout + j];
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
            m[r*cols + c] = expf(m[r*cols + c] - maxv);
            sum += m[r*cols + c];
        }
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] /= sum;
        }
    }
}

/* ============================================================
 * layer_norm: normalize each token vector (row) to zero mean, unit variance
 *   y[t][d] = (x[t][d] - mean_t) / sqrt(var_t + eps) * gamma[d] + beta[d]
 *   Applied per token (row), not per feature (column).
 *   input/output [T][D]
 * ============================================================ */
static void layer_norm(const float *x, float *y,
                       const float *gamma, const float *beta,
                       int T, int D, float eps)
{
    for (int t = 0; t < T; t++) {               /* t: one token = one row */
        /* compute mean over feature dimension */
        float mean = 0.0f;
        for (int d = 0; d < D; d++) {
            mean += x[t*D + d];                 /* sum features for token t */
        }
        mean /= (float)D;                       /* mean over D features */

        /* compute variance */
        float var = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = x[t*D + d] - mean;     /* deviation from mean */
            var += diff * diff;                 /* squared deviation */
        }
        var /= (float)D;                        /* variance over D features */

        float inv_std = 1.0f / sqrtf(var + eps); /* 1/sqrt(var + eps): avoids divide-by-zero */

        /* normalize, scale, shift */
        for (int d = 0; d < D; d++) {
            y[t*D + d] = ((x[t*D + d] - mean) * inv_std) /* normalize to N(0,1) */
                         * gamma[d]                        /* learned scale per feature */
                         + beta[d];                        /* learned shift per feature */
        }
    }
}

/* ============================================================
 * self_attn_4d: multi-head self-attention on [N][T][D] input
 *   4D internal layout: [N][H][T][Dh]
 *   out: [N][T][D]
 * ============================================================ */
static void self_attn_4d(
    const float *x,    /* [N][T][D] */
    const float *Wq, const float *Wk, const float *Wv, const float *Wo,
    float *out,        /* [N][T][D] */
    int N, int H, int T, int D, int Dh)
{
    int HD = H * Dh;

    /* T is dynamic — sequence length varies per sample in VLA inference */
    int stride_T_4d = Dh;
    int stride_H_4d = T * Dh;   /* depends on runtime T — computed at call time */
    int stride_N_4d = H * T * Dh;

    float scale = 1.0f / sqrtf((float)Dh); /* 1/sqrt(Dh) attention scale */

    float *Q4d   = calloc(N*H*T*Dh, sizeof(float));
    float *K4d   = calloc(N*H*T*Dh, sizeof(float));
    float *V4d   = calloc(N*H*T*Dh, sizeof(float));
    float *aout  = calloc(N*H*T*Dh, sizeof(float));
    float *cbuf  = malloc(T * HD   * sizeof(float));
    float *scores= malloc(T * T    * sizeof(float));
    float *tmpP  = malloc(T * HD   * sizeof(float));

    for (int n = 0; n < N; n++) {
        /* project input to Q, K, V */
        matmul(x + n*T*D, Wq, tmpP, T, D, HD); /* Q = input @ W_q */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    Q4d[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d] =
                        tmpP[t*HD + h*Dh + d]; /* reshape to 4D head-first */
                }
            }
        }
        matmul(x + n*T*D, Wk, tmpP, T, D, HD);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    K4d[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d] =
                        tmpP[t*HD + h*Dh + d];
                }
            }
        }
        matmul(x + n*T*D, Wv, tmpP, T, D, HD);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    V4d[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d] =
                        tmpP[t*HD + h*Dh + d];
                }
            }
        }

        /* SDPA per head */
        for (int h = 0; h < H; h++) {
            const float *Qh = Q4d + n*stride_N_4d + h*stride_H_4d;
            const float *Kh = K4d + n*stride_N_4d + h*stride_H_4d;
            const float *Vh = V4d + n*stride_N_4d + h*stride_H_4d;
            float       *Oh = aout + n*stride_N_4d + h*stride_H_4d;

            for (int tq = 0; tq < T; tq++) {
                for (int tk = 0; tk < T; tk++) {
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; d++) {
                        dot += Qh[tq*Dh + d] * Kh[tk*Dh + d];
                    }
                    scores[tq*T + tk] = dot * scale;
                }
            }
            softmax_rows(scores, T, T);
            for (int tq = 0; tq < T; tq++) {
                for (int d = 0; d < Dh; d++) {
                    float acc = 0.0f;
                    for (int tk = 0; tk < T; tk++) {
                        acc += scores[tq*T + tk] * Vh[tk*Dh + d];
                    }
                    Oh[tq*Dh + d] = acc;
                }
            }
        }

        /* reshape [H][T][Dh] → [T][H*Dh] → W_o → [T][D] */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < Dh; d++) {
                    cbuf[t*HD + h*Dh + d] =
                        aout[n*stride_N_4d + h*stride_H_4d + t*stride_T_4d + d];
                }
            }
        }
        matmul(cbuf, Wo, out + n*T*D, T, HD, D);
    }

    free(Q4d); free(K4d); free(V4d); free(aout); free(cbuf); free(scores); free(tmpP);
}

/* ============================================================
 * mhsa_block_prenorm: x + SelfAttention(LayerNorm(x))
 *   Pre-norm variant: normalize BEFORE attention — more stable for deep models.
 *   Used in ViT-L (OpenVLA backbone).
 * ============================================================ */
static void mhsa_block_prenorm(
    const float *x,     /* [N][T][D] input */
    float *out,         /* [N][T][D] output */
    float *ln_out,      /* scratch [N][T][D] for layernorm output */
    float *attn_out,    /* scratch [N][T][D] for attention output */
    const float *Wq, const float *Wk, const float *Wv, const float *Wo,
    const float *gamma, const float *beta,
    int N, int H, int T, int D, int Dh)
{
    /* Step 1: LayerNorm(x) — normalize each token before attention */
    for (int n = 0; n < N; n++) {
        layer_norm(x + n*T*D, ln_out + n*T*D, gamma, beta, T, D, 1e-5f);
    }

    /* Step 2: SelfAttention(LayerNorm(x)) — attention on normalized input */
    self_attn_4d(ln_out, Wq, Wk, Wv, Wo, attn_out, N, H, T, D, Dh);

    /* Step 3: residual add: out = x + attn(ln(x))
     * Residual ensures gradients flow even if attention learns identity. */
    for (int i = 0; i < N*T*D; i++) {
        out[i] = x[i] + attn_out[i]; /* add original input back: skip connection */
    }
}

/* ============================================================
 * Demo: N=1 H=2 T=3 D=8 Dh=4
 * Show: input → after attention → after residual → after layernorm
 * ============================================================ */
int main(void)
{
    int N = 1, H = 2, T = 3, D = 8, Dh = 4;
    int HD = H * Dh; /* = 8 = D */

    float *x        = calloc(N*T*D, sizeof(float));
    float *Wq       = calloc(D*HD,  sizeof(float));
    float *Wk       = calloc(D*HD,  sizeof(float));
    float *Wv       = calloc(D*HD,  sizeof(float));
    float *Wo       = calloc(HD*D,  sizeof(float));
    float *gamma    = malloc(D * sizeof(float));
    float *beta     = calloc(D, sizeof(float));
    float *ln_out   = calloc(N*T*D, sizeof(float));
    float *attn_out = calloc(N*T*D, sizeof(float));
    float *out      = calloc(N*T*D, sizeof(float));
    float *post_ln  = calloc(N*T*D, sizeof(float)); /* final post-norm output for display */

    /* init gamma = 1 (identity scale initially) */
    for (int d = 0; d < D; d++) { gamma[d] = 1.0f; }

    /* input: diagonal-ish pattern to make residual connection visible */
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            x[t*D + d] = (float)(t + 1) * (d == t ? 2.0f : 0.1f); /* peak at diagonal */
        }
    }

    /* identity projections */
    for (int i = 0; i < D; i++) {
        Wq[i*HD + i] = 1.0f;
        Wk[i*HD + i] = 1.0f;
        Wv[i*HD + i] = 1.0f;
        Wo[i*D  + i] = 1.0f;
    }

    printf("=== Full MHSA Block (Pre-Norm) Demo ===\n");
    printf("N=%d H=%d T=%d D=%d Dh=%d\n", N, H, T, D, Dh);
    printf("Pre-norm: out = x + SelfAttention(LayerNorm(x))\n");
    printf("MHSA block = attention + residual + norm.\n");
    printf("Residual (add x back): ensures gradients flow even if attention learns identity.\n");
    printf("Pre-norm: norm before attention — more stable for deep models — used in ViT-L (OpenVLA backbone).\n\n");

    print_grid("Input x  [T x D]", x, T, D);

    /* show after layernorm (before attention) */
    layer_norm(x, ln_out, gamma, beta, T, D, 1e-5f);
    print_grid("After LayerNorm(x)  [T x D]", ln_out, T, D);

    /* run self-attention on ln_out */
    self_attn_4d(ln_out, Wq, Wk, Wv, Wo, attn_out, N, H, T, D, Dh);
    print_grid("After SelfAttention(LayerNorm(x))  [T x D]", attn_out, T, D);

    /* residual add */
    mhsa_block_prenorm(x, out, ln_out, attn_out,
                       Wq, Wk, Wv, Wo, gamma, beta,
                       N, H, T, D, Dh);

    print_grid("After residual add (x + attention)  [T x D]", out, T, D);

    /* optional final layernorm (post-block) for display */
    layer_norm(out, post_ln, gamma, beta, T, D, 1e-5f);
    print_grid("After final LayerNorm (post-block)  [T x D]", post_ln, T, D);

    /* checks */
    printf("--- Checks ---\n");
    /* output shape same as input */
    printf("  Output shape [%d][%d] matches input shape [%d][%d]: PASS\n", T, D, T, D);

    /* output norms finite */
    int ok = 1;
    for (int t = 0; t < T; t++) {
        float norm = 0.0f;
        for (int d = 0; d < D; d++) {
            norm += out[t*D + d] * out[t*D + d];
        }
        norm = sqrtf(norm);
        int pass = isfinite(norm) && norm > 0.0f;
        printf("  token=%d output norm=%.4f  %s\n", t, norm, pass ? "PASS" : "FAIL");
        if (!pass) { ok = 0; }
    }
    printf("[04] MHSA Full Block %s\n", ok ? "PASS" : "FAIL");

    free(x); free(Wq); free(Wk); free(Wv); free(Wo);
    free(gamma); free(beta); free(ln_out); free(attn_out); free(out); free(post_ln);
    return 0;
}
