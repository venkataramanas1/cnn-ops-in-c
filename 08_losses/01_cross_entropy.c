/* ============================================================================
 * [01] Cross-Entropy Loss  --  Multi-class Classification
 * ----------------------------------------------------------------------------
 * CE loss penalizes wrong confident predictions exponentially. The log(p) term:
 * if p→0 for the true class, loss→∞. Used in ViT/VLA classification heads.
 *
 * Formula per sample:
 *   softmax: p[c] = exp(z[c]) / sum_k exp(z[k])
 *   L = -sum_c [ y[c] * log(p[c]) ]   (y is one-hot, so only true class matters)
 *   averaged over batch N
 *
 * Layout:
 *   logits  [N][C]  -- N samples, C classes
 *   labels  [N][C]  -- one-hot
 *   4D ext  [N][T][C]  -- sequence labeling; stride shown below
 *
 * Build: gcc 01_cross_entropy.c -o 01_cross_entropy -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---- print a 1-D float array with a label --------------------------------- */
static void show_vec(const char *title, const float *v, int n)
{
    printf("%s  [N=%d]\n", title, n);
    printf("   ");
    for (int i = 0; i < n; i++) {
        printf("%8.4f", v[i]);
    }
    printf("\n\n");
}

/* ---- print a 2-D float matrix [N][C] -------------------------------------- */
static void show_mat(const char *title, const float *m, int N, int C)
{
    printf("%s  [N=%d C=%d]\n", title, N, C);
    for (int n = 0; n < N; n++) {
        printf("  sample %d: ", n);
        for (int c = 0; c < C; c++) {
            printf("%8.4f", m[n * C + c]);   /* stride: row = n*C */
        }
        printf("\n");
    }
    printf("\n");
}

/* ---- softmax in-place over class dimension for one sample [C] ------------- */
static void softmax(float *z, float *p, int C)
{
    /* step 1: find max for numerical stability (subtract before exp) */
    float mx = z[0];
    for (int c = 1; c < C; c++) {
        if (z[c] > mx) { mx = z[c]; }
    }

    /* step 2: exp(z[c] - max) so the largest value becomes exp(0)=1 */
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        p[c] = expf(z[c] - mx);          /* shifted exp */
        sum += p[c];
    }

    /* step 3: normalize so probabilities sum to 1 */
    for (int c = 0; c < C; c++) {
        p[c] /= sum;
    }
}

/* ---- cross-entropy loss [N][C] logits, [N][C] one-hot labels -------------- */
/* returns mean loss over N samples; also fills per_sample[N] */
static float cross_entropy(const float *logits, const float *labels,
                            float *per_sample, int N, int C)
{
    float total = 0.0f;
    float *p = malloc(C * sizeof(float));   /* softmax probs for one sample */

    for (int n = 0; n < N; n++) {
        /* pointer to this sample's logits row: stride = C */
        const float *z_n = logits + n * C;

        /* mutable copy for softmax (softmax writes to p) */
        float *z_copy = malloc(C * sizeof(float));
        for (int c = 0; c < C; c++) {
            z_copy[c] = z_n[c];
        }

        softmax(z_copy, p, C);             /* compute p[c] = softmax(z)[c] */

        /* L_n = -sum_c y[c] * log(p[c])
         * y is one-hot so only the true class contributes */
        float loss_n = 0.0f;
        for (int c = 0; c < C; c++) {
            if (labels[n * C + c] > 0.5f) {        /* true class (one-hot=1) */
                /* clamp p to avoid log(0) */
                float p_clamp = p[c] < 1e-9f ? 1e-9f : p[c];
                loss_n -= logf(p_clamp);            /* -= log(p_true) */
            }
        }
        per_sample[n] = loss_n;
        total += loss_n;
        free(z_copy);
    }

    free(p);
    return total / (float)N;              /* mean over batch */
}

/* ---- 4D extension: [N][T][C] sequence labeling ----------------------------
 * stride computation:
 *   element [n][t][c] = base + n*(T*C) + t*C + c
 * Each (n,t) position is an independent softmax+CE, then average over N*T.
 * --------------------------------------------------------------------- */
static float cross_entropy_4d(const float *logits, const float *labels,
                               int N, int T, int C)
{
    int stride_n = T * C;    /* how many floats to skip to advance one sample */
    int stride_t = C;        /* how many floats to skip to advance one timestep */

    float total = 0.0f;
    int count   = 0;
    float *z_copy = malloc(C * sizeof(float));
    float *p      = malloc(C * sizeof(float));

    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            /* base address for position (n, t) */
            const float *z_nt = logits + n * stride_n + t * stride_t;
            const float *y_nt = labels + n * stride_n + t * stride_t;

            for (int c = 0; c < C; c++) {
                z_copy[c] = z_nt[c];
            }
            softmax(z_copy, p, C);

            float loss_nt = 0.0f;
            for (int c = 0; c < C; c++) {
                if (y_nt[c] > 0.5f) {
                    float p_clamp = p[c] < 1e-9f ? 1e-9f : p[c];
                    loss_nt -= logf(p_clamp);
                }
            }
            total += loss_nt;
            count++;
        }
    }

    free(z_copy);
    free(p);
    return total / (float)count;
}

int main(void)
{
    /* ---- 2D demo: N=2 samples, C=3 classes -------------------------------- */
    int N = 2, C = 3;

    /* sample 0: confident-correct  -- logit for class 0 is very high */
    /* sample 1: confident-wrong    -- logit for class 2 is high, but true class is 0 */
    float logits[2 * 3] = {
        10.0f,  0.0f,  0.0f,   /* sample 0: model says class 0 strongly */
         0.0f,  0.0f, 10.0f    /* sample 1: model says class 2 strongly */
    };

    /* one-hot labels: both samples have true class = 0 */
    float labels[2 * 3] = {
        1.0f, 0.0f, 0.0f,      /* sample 0 true class = 0 */
        1.0f, 0.0f, 0.0f       /* sample 1 true class = 0 */
    };

    show_mat("LOGITS  [N][C]",  logits, N, C);
    show_mat("LABELS  [N][C] (one-hot)", labels, N, C);

    float per_sample[2];
    float mean_loss = cross_entropy(logits, labels, per_sample, N, C);

    show_vec("PER-SAMPLE LOSS", per_sample, N);
    printf("mean CE loss = %.4f\n\n", mean_loss);

    /* check */
    printf("[01] check: sample0 (confident-correct) loss = %.4f (expected ≈ 0)\n",
           per_sample[0]);
    printf("[01] check: sample1 (confident-wrong)   loss = %.4f (expected >> 0)\n\n",
           per_sample[1]);

    /* ---- 4D extension demo: N=1 T=2 C=3 ---------------------------------- */
    int T = 2;
    printf("--- 4D extension [N=%d][T=%d][C=%d], stride_n=%d stride_t=%d ---\n",
           N, T, C, T * C, C);

    /* reuse same logits/labels for 2 timesteps of one sample */
    float logits_4d[1 * 2 * 3] = {
        10.0f, 0.0f, 0.0f,     /* n=0, t=0: confident-correct */
         0.0f, 0.0f, 10.0f     /* n=0, t=1: confident-wrong */
    };
    float labels_4d[1 * 2 * 3] = {
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f
    };

    float loss_4d = cross_entropy_4d(logits_4d, labels_4d, 1, T, C);
    printf("4D mean CE loss = %.4f  (should equal mean of [≈0, >>0])\n\n", loss_4d);

    return 0;
}
