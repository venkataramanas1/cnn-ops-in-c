/*
 * 02_gru_cell.c
 *
 * Op: GRU time step — simpler than LSTM (no separate cell state).
 *
 * GRU: update gate z blends old h with new candidate — z=0 keeps old state,
 * z=1 replaces it. Reset gate r gates how much of the old hidden state
 * influences the candidate. Fewer parameters than LSTM.
 *
 * Per-step gate equations (all [N][D_h]):
 *   combined_full = [x_t | h_t]               concat, dim = D_in + D_h
 *   z = sigmoid(W_z @ combined_full + b_z)    update gate -- blend ratio
 *   r = sigmoid(W_r @ combined_full + b_r)    reset gate  -- filter old h
 *   combined_r   = [x_t | r * h_t]           gated concat for candidate
 *   h_tilde = tanh(W_h @ combined_r + b_h)   candidate hidden state
 *   h_new = (1-z)*h_t + z*h_tilde            interpolate old and new
 *
 * Weight layout: W_* [D_h][D_in+D_h], flat: dh*(D_in+D_h) + di
 * Input  layout: x   [N][T][D_in],    flat: n*T*D_in + t*D_in + d
 * Hidden layout: h   [N][D_h],        flat: n*D_h + d
 *
 * 4D extension: wrap cell over T steps → full sequence [N][T][D_in] → [N][T][D_h]
 *
 * WHERE IN MODELS:
 *   - GRU-based VLA edge models for on-device action policies.
 *   - Lightweight audio/speech models where latency matters.
 *   - Any setting where LSTM parameter count is prohibitive.
 *
 * Build:
 *   gcc -O2 -o 02_gru_cell 02_gru_cell.c -lm && ./02_gru_cell
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

/* Demo dimensions — same as LSTM for direct comparison */
#define N    1    /* batch size            */
#define D_IN 2    /* input feature dim     */
#define D_H  2    /* hidden state dim      */
#define T    3    /* sequence length (demo); in real code T is a runtime param */

#define D_C  (D_IN + D_H)  /* combined dim for concat */

/* ------------------------------------------------------------------ */
/* elementwise helpers                                                  */
/* ------------------------------------------------------------------ */

static float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));  /* squash to (0,1) */
}

static float tanh_f(float x)
{
    return tanhf(x);  /* squash to (-1,1) */
}

/* ------------------------------------------------------------------ */
/* gru_step                                                             */
/*                                                                      */
/* Runs one GRU time step for all N samples.                           */
/*                                                                      */
/* Dynamic shape note:                                                  */
/*   T is a runtime parameter — sequence length not known at compile   */
/*   time. stride_t = D_in is the step along the time axis.            */
/* ------------------------------------------------------------------ */
static void gru_step(
    const float *x_t,   /* [N][D_in] — current input token           */
    const float *h_t,   /* [N][D_h]  — prev hidden state              */
    const float *W_z,   /* [D_h][D_c] — update gate weights           */
    const float *W_r,   /* [D_h][D_c] — reset  gate weights           */
    const float *W_h,   /* [D_h][D_c] — candidate hidden weights      */
    const float *b_z,   /* [D_h] — update gate bias                   */
    const float *b_r,   /* [D_h] — reset  gate bias                   */
    const float *b_h,   /* [D_h] — candidate hidden bias              */
    float       *h_new, /* [N][D_h] — new hidden state (output)       */
    int d_in, int d_h, int n_dim)
{
    int n, dh, dc;
    int d_c = d_in + d_h;  /* combined dimension */

    for (n = 0; n < n_dim; n++)
    {
        /* Step 1: build combined_full = [x_t[n] | h_t[n]] for z and r gates */
        float combined_full[D_C];  /* fixed-size for demo */
        for (dc = 0; dc < d_in; dc++)
        {
            combined_full[dc] = x_t[n * d_in + dc];        /* x portion */
        }
        for (dc = 0; dc < d_h; dc++)
        {
            combined_full[d_in + dc] = h_t[n * d_h + dc];  /* h portion */
        }

        /* Step 2: compute update gate z and reset gate r */
        float z[D_H];  /* update gate values per hidden dim */
        float r[D_H];  /* reset gate values per hidden dim  */

        for (dh = 0; dh < d_h; dh++)
        {
            float pre_z = b_z[dh];
            float pre_r = b_r[dh];

            for (dc = 0; dc < d_c; dc++)
            {
                /* W[dh][dc]: flat dh*d_c + dc */
                pre_z += W_z[dh * d_c + dc] * combined_full[dc];
                pre_r += W_r[dh * d_c + dc] * combined_full[dc];
            }

            z[dh] = sigmoid_f(pre_z);  /* z=0: keep old, z=1: replace */
            r[dh] = sigmoid_f(pre_r);  /* r gates how much old h matters */
        }

        /* Step 3: build combined_r = [x_t | r * h_t] for candidate */
        float combined_r[D_C];
        for (dc = 0; dc < d_in; dc++)
        {
            combined_r[dc] = x_t[n * d_in + dc];               /* x unchanged */
        }
        for (dc = 0; dc < d_h; dc++)
        {
            /* reset gate scales how much previous hidden state leaks in */
            combined_r[d_in + dc] = r[dc] * h_t[n * d_h + dc];
        }

        /* Step 4: compute candidate h_tilde and blend with old h */
        for (dh = 0; dh < d_h; dh++)
        {
            float pre_h = b_h[dh];

            for (dc = 0; dc < d_c; dc++)
            {
                pre_h += W_h[dh * d_c + dc] * combined_r[dc];
            }

            float h_tilde = tanh_f(pre_h);  /* candidate: new state proposal */

            /* blend: z=0 → keep h_t entirely; z=1 → replace with h_tilde */
            h_new[n * d_h + dh] = (1.0f - z[dh]) * h_t[n * d_h + dh]
                                 + z[dh] * h_tilde;
        }
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    int t, d;

    printf("=== 02_gru_cell ===\n");
    printf("GRU: update gate z blends old h with new candidate.\n");
    printf("z=0 keeps old state, z=1 replaces it. Reset gate r filters old h.\n");
    printf("Fewer parameters than LSTM (no separate cell state).\n\n");

    /*
     * T is dynamic: sequence length not known at compile time.
     * stride_t = D_in is the step along the time axis.
     */
    int seq_len = T;
    int stride_t = D_IN;  /* step along time axis */

    printf("Sequence [N=%d][T=%d][D_in=%d]: (same as LSTM demo for comparison)\n",
           N, seq_len, D_IN);
    printf("(T is dynamic runtime param)\n");

    /* Same input sequence as LSTM demo: [[1,0],[0,1],[1,1]] */
    float x_seq[N * T * D_IN] = {
        1.0f, 0.0f,   /* t=0 */
        0.0f, 1.0f,   /* t=1 */
        1.0f, 1.0f    /* t=2 */
    };

    for (t = 0; t < seq_len; t++)
    {
        printf("  t=%d: [", t);
        for (d = 0; d < D_IN; d++)
        {
            /* x_seq[n=0][t][d]: flat n*T*D_in + t*stride_t + d */
            printf(" %.1f", x_seq[0 * T * D_IN + t * stride_t + d]);
        }
        printf(" ]\n");
    }

    /*
     * Weights: same identity-like pattern as LSTM (0.5 on diagonal).
     * W shape [D_h][D_c] where D_c = D_in + D_h = 4.
     * row0 = [0.5, 0,   0.5, 0  ]
     * row1 = [0,   0.5, 0,   0.5]
     */
    float W_z[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.5f
    };
    float W_r[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.5f
    };
    float W_h[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.5f
    };

    /* biases = 0 */
    float b_z[D_H] = {0.0f, 0.0f};
    float b_r[D_H] = {0.0f, 0.0f};
    float b_h[D_H] = {0.0f, 0.0f};

    /* initial hidden state = zeros */
    float h[N * D_H] = {0.0f, 0.0f};
    float h_new[N * D_H];

    printf("\nUnrolling GRU over T=%d steps:\n", seq_len);
    printf("(initial h=[0,0])\n\n");

    /* t: time step — GRU unrolled over dynamic T */
    for (t = 0; t < seq_len; t++)
    {
        /* x_t is a time slice of x_seq: flat offset t*stride_t for n=0 */
        const float *x_t = x_seq + 0 * T * D_IN + t * stride_t;

        gru_step(x_t, h,
                 W_z, W_r, W_h,
                 b_z, b_r, b_h,
                 h_new,
                 D_IN, D_H, N);

        memcpy(h, h_new, N * D_H * sizeof(float));

        printf("  t=%d  h=[", t);
        for (d = 0; d < D_H; d++)
        {
            printf(" %7.4f", h[d]);
        }
        printf(" ]\n");
    }

    /* Check: all h values in (-1,1) — tanh bounded output */
    printf("\nCHECK: all h values in (-1,1) from tanh bound:\n");
    int all_ok = 1;
    for (d = 0; d < D_H; d++)
    {
        int ok = (h[d] > -1.0f && h[d] < 1.0f);
        printf("  h[%d] = %7.4f  %s\n", d, h[d], ok ? "in (-1,1) OK" : "OUT OF RANGE!");
        if (!ok)
        {
            all_ok = 0;
        }
    }
    printf("RESULT: %s\n", all_ok ? "PASS" : "FAIL");

    return 0;
}
