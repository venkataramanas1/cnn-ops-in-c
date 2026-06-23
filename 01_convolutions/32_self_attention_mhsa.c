/* ============================================================================
 * [32] Self-Attention (Multi-Head Self-Attention, MHSA)
 * ----------------------------------------------------------------------------
 * SHAPE: NO local spatial kernel. Every position looks at every other position
 * simultaneously. From the top: a complete web of connections between every
 * pixel and every other pixel -- fully global, no locality. Used in Vision
 * Transformers (ViT) and many hybrid CNN-Transformer architectures.
 *
 * For a sequence of N tokens, each of dimension D:
 *   Input : [N][D]
 *   Wq, Wk, Wv: [D][D_head]  per head  (split into H heads)
 *   Q = X @ Wq,  K = X @ Wk,  V = X @ Wv    [N][D_head]
 *   Attn = softmax(Q @ K^T / sqrt(D_head))   [N][N]   <- the web of weights
 *   Head = Attn @ V                           [N][D_head]
 *   Output = concat(heads) @ Wo              [N][D]
 *
 *   Input : [N][D]
 *   Weight: Wq,Wk,Wv:[D][D_head] x H heads;  Wo:[D][D]
 *   Output: [N][D]
 *
 * FLOPs ~ 4 * N * D^2 + 2 * N^2 * D    (the N^2 term is the attention web)
 *
 * DEMO: N=4 tokens, D=4, single head. Print Q, K, attention weights, V, output.
 * Each token attends to ALL others -- watch the attention matrix.
 *
 * Build:  gcc 32_self_attention_mhsa.c -o 32_self_attention_mhsa -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void show_mat(const char *title, const float *m, int rows, int cols)
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

/* matrix multiply: C = A (M x K) @ B (K x N)
 * Core building block for Q/K/V projections and the final weighted sum of V. */
static void matmul(const float *A, const float *B, float *C, int M, int Kd, int N)
{
    for (int i = 0; i < M; i++) {        /* i: row of A / row of output C */
        for (int j = 0; j < N; j++) {   /* j: col of B / col of output C */
            float s = 0.0f;
            for (int k = 0; k < Kd; k++) { /* k: shared contraction dimension */
                /* A flat index: [i][k] = i*Kd + k;  B flat index: [k][j] = k*N + j */
                s += A[i*Kd + k] * B[k*N + j];
            }
            C[i*N + j] = s;             /* C flat index: [i][j] = i*N + j */
        }
    }
}

/* softmax along last dimension (rows): converts raw attention scores to probabilities
 * subtract maxv first for numerical stability (does not change output but prevents overflow) */
static void softmax_rows(float *m, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {     /* r: token (query) index -- one softmax per row */
        /* find row max for numerically stable exp */
        float maxv = m[r*cols];
        for (int c = 1; c < cols; c++) {
            if (m[r*cols + c] > maxv) { maxv = m[r*cols + c]; }
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {         /* c: key token being attended to */
            m[r*cols + c] = expf(m[r*cols + c] - maxv); /* shift by max before exp */
            sum += m[r*cols + c];
        }
        /* normalize so each row sums to 1 (attention weights are a probability distribution) */
        for (int c = 0; c < cols; c++) {
            m[r*cols + c] /= sum;
        }
    }
}

/* transpose: B[N][M] = A[M][N]^T -- needed to form K^T for the Q @ K^T score matrix */
static void transpose(const float *A, float *B, int M, int N)
{
    for (int i = 0; i < M; i++) {        /* i: row in A */
        for (int j = 0; j < N; j++) {   /* j: col in A */
            B[j*M + i] = A[i*N + j];    /* A[i][j] -> B[j][i]: row/col swap */
        }
    }
}

int main(void)
{
    int N = 4, D = 4, Dh = 4;   /* N tokens, D dim, single head of size Dh=D */

    float *X  = calloc(N*D,  sizeof(float));
    float *Wq = calloc(D*Dh, sizeof(float));
    float *Wk = calloc(D*Dh, sizeof(float));
    float *Wv = calloc(D*Dh, sizeof(float));
    float *Q  = calloc(N*Dh, sizeof(float));
    float *K  = calloc(N*Dh, sizeof(float));
    float *V  = calloc(N*Dh, sizeof(float));
    float *Kt = calloc(Dh*N, sizeof(float));
    float *A  = calloc(N*N,  sizeof(float));
    float *out= calloc(N*Dh, sizeof(float));

    /* tokens: each token is a one-hot indicator [1,0,0,0], [0,2,0,0], ... */
    X[0*D + 0] = 1.0f;
    X[1*D + 1] = 2.0f;
    X[2*D + 2] = 3.0f;
    X[3*D + 3] = 4.0f;

    /* identity projections so Q=K=V=X for clarity */
    for (int i = 0; i < D; i++) {
        Wq[i*Dh + i] = 1.0f;
        Wk[i*Dh + i] = 1.0f;
        Wv[i*Dh + i] = 1.0f;
    }

    /* step 1: Q/K/V projections -- each token is projected into three separate spaces:
     *   Q (query): "what am I looking for?"
     *   K (key):   "what do I contain?"
     *   V (value): "what information do I carry?" */
    matmul(X, Wq, Q, N, D, Dh);  /* Q = X @ Wq  [N x Dh]: query projections */
    matmul(X, Wk, K, N, D, Dh);  /* K = X @ Wk  [N x Dh]: key projections */
    matmul(X, Wv, V, N, D, Dh);  /* V = X @ Wv  [N x Dh]: value projections */

    show_mat("INPUT X  (4 tokens, one-hot scaled)", X, N, D);
    show_mat("Q (queries)", Q, N, Dh);
    show_mat("K (keys)",    K, N, Dh);
    show_mat("V (values)",  V, N, Dh);

    /* step 2: compute attention scores = Q @ K^T / sqrt(Dh)
     *   Q @ K^T gives an [N x N] matrix: entry [i][j] = how much token i attends to token j
     *   dividing by sqrt(Dh) prevents dot products from growing too large (keeps softmax gradients healthy) */
    transpose(K, Kt, N, Dh);      /* Kt = K^T  [Dh x N]: transpose keys for matrix multiply */
    matmul(Q, Kt, A, N, Dh, N);   /* A = Q @ K^T  [N x N]: raw attention score matrix */

    float scale = 1.0f / sqrtf((float)Dh); /* scale factor 1/sqrt(Dh) */
    for (int i = 0; i < N*N; i++) { A[i] *= scale; } /* scale all scores: score = Q*K^T / sqrt(Dh) */
    /* step 3: softmax over each row: converts scores to probability distribution over keys */
    softmax_rows(A, N, N);        /* A[i][j] now = attention weight token i assigns to token j */

    show_mat("ATTENTION weights (softmax(Q @ K^T / sqrt(Dh)))", A, N, N);

    /* step 4: weighted sum of V -- each output token is a blend of all value vectors
     * weighted by the attention probabilities: out[i] = sum_j( A[i][j] * V[j] ) */
    matmul(A, V, out, N, N, Dh);  /* out = A @ V  [N x Dh]: context-aware token representations */
    show_mat("OUTPUT = Attention @ V", out, N, Dh);

    /* each token should mostly attend to itself (diagonal dominant) */
    printf("[32] check: token0 attends most to itself: A[0][0]=%.4f\n", A[0]);

    free(X);free(Wq);free(Wk);free(Wv);free(Q);free(K);free(V);
    free(Kt);free(A);free(out);
    return 0;
}
