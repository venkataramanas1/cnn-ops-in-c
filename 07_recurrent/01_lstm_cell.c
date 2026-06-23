/*
 * 01_lstm_cell.c
 *
 * Op: Single LSTM time step on a batch of sequences, then unrolled over T steps.
 *
 * LSTM cell: the forget gate f learns what to ERASE from cell state,
 * input gate i what to WRITE, output gate o what to READ.
 * The cell state c is the 'long-term memory' that gradients flow through.
 *
 * Per-step gate equations (all [N][D_h]):
 *   combined = [x_t | h_t]             concat, dim = D_in + D_h
 *   i = sigmoid(W_i @ combined + b_i)  input gate  -- what to WRITE
 *   f = sigmoid(W_f @ combined + b_f)  forget gate -- what to ERASE
 *   g = tanh   (W_g @ combined + b_g)  cell gate   -- candidate values
 *   o = sigmoid(W_o @ combined + b_o)  output gate -- what to READ
 *   c_new = f * c_t + i * g            cell update (long-term memory)
 *   h_new = o * tanh(c_new)            hidden update (short-term output)
 *
 * Weight layout: W_* [D_h][D_in+D_h], flat: dh*(D_in+D_h) + di
 * Input  layout: x   [N][T][D_in],    flat: n*T*D_in + t*D_in + d
 * Hidden layout: h   [N][D_h],        flat: n*D_h + d
 * Cell   layout: c   [N][D_h],        flat: n*D_h + d
 *
 * 4D extension: wrap cell over T steps → full sequence [N][T][D_in] → [N][T][D_h]
 *
 * WHERE IN MODELS:
 *   - RT-1 uses FiLM+LSTM for temporal context in robot action decoding.
 *   - Lightweight VLA edge models use LSTM for sequence history.
 *   - Octo uses transformer, but many embedded/MCU variants fall back to LSTM.
 *
 * Build:
 *   gcc -O2 -o 01_lstm_cell 01_lstm_cell.c -lm && ./01_lstm_cell
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

/* Demo dimensions */
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
/* lstm_step                                                            */
/*                                                                      */
/* Runs one time step for all N samples in the batch.                  */
/*                                                                      */
/* Dynamic shape note:                                                  */
/*   T is a runtime parameter — sequence length not known at compile   */
/*   time. Inside the T-loop each call to lstm_step sees a slice of    */
/*   the sequence; stride_t = D_in is the step along the time axis.    */
/* ------------------------------------------------------------------ */
static void lstm_step(
    const float *x_t,   /* [N][D_in] — current input token          */
    const float *h_t,   /* [N][D_h]  — prev hidden state             */
    const float *c_t,   /* [N][D_h]  — prev cell state               */
    const float *W_i,   /* [D_h][D_c] — input  gate weights          */
    const float *W_f,   /* [D_h][D_c] — forget gate weights          */
    const float *W_g,   /* [D_h][D_c] — cell   gate weights          */
    const float *W_o,   /* [D_h][D_c] — output gate weights          */
    const float *b_i,   /* [D_h] — input  gate bias                  */
    const float *b_f,   /* [D_h] — forget gate bias                  */
    const float *b_g,   /* [D_h] — cell   gate bias                  */
    const float *b_o,   /* [D_h] — output gate bias                  */
    float       *h_new, /* [N][D_h] — new hidden state (output)      */
    float       *c_new, /* [N][D_h] — new cell state (output)        */
    int d_in, int d_h, int n_dim)
{
    int n, dh, dc;
    int d_c = d_in + d_h;  /* combined dimension for concat */

    for (n = 0; n < n_dim; n++)
    {
        /* Step 1: build combined = [x_t[n] | h_t[n]] into a local buffer */
        float combined[D_C];  /* fixed-size for demo; use malloc for dynamic */
        for (dc = 0; dc < d_in; dc++)
        {
            combined[dc] = x_t[n * d_in + dc];       /* first D_in slots = x */
        }
        for (dc = 0; dc < d_h; dc++)
        {
            combined[d_in + dc] = h_t[n * d_h + dc]; /* next D_h slots = h   */
        }

        /* Step 2: compute each gate via W @ combined + b, then activation */
        for (dh = 0; dh < d_h; dh++)
        {
            /* accumulate: pre_gate = b + sum_dc W[dh][dc] * combined[dc] */
            float pre_i = b_i[dh];
            float pre_f = b_f[dh];
            float pre_g = b_g[dh];
            float pre_o = b_o[dh];

            for (dc = 0; dc < d_c; dc++)
            {
                /* W[dh][dc]: flat index dh*d_c + dc */
                pre_i += W_i[dh * d_c + dc] * combined[dc];
                pre_f += W_f[dh * d_c + dc] * combined[dc];
                pre_g += W_g[dh * d_c + dc] * combined[dc];
                pre_o += W_o[dh * d_c + dc] * combined[dc];
            }

            /* apply activations */
            float i_gate = sigmoid_f(pre_i);   /* input gate: what to WRITE */
            float f_gate = sigmoid_f(pre_f);   /* forget gate: what to ERASE */
            float g_gate = tanh_f(pre_g);      /* cell gate: candidate values */
            float o_gate = sigmoid_f(pre_o);   /* output gate: what to READ   */

            /* cell update: blend old cell (scaled by f) with new candidate (scaled by i) */
            float c_val = f_gate * c_t[n * d_h + dh]  /* forget old cell */
                        + i_gate * g_gate;             /* write new candidate */

            /* hidden update: read from cell through output gate */
            float h_val = o_gate * tanh_f(c_val);

            /* write outputs: flat n*d_h + dh */
            c_new[n * d_h + dh] = c_val;
            h_new[n * d_h + dh] = h_val;
        }
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    int t, d;

    printf("=== 01_lstm_cell ===\n");
    printf("LSTM cell: forget gate f=ERASE, input gate i=WRITE, output gate o=READ.\n");
    printf("Cell state c is long-term memory; h is the step output.\n\n");

    /*
     * T is dynamic: sequence length not known at compile time.
     * In production, T is passed as a runtime parameter.
     */
    int seq_len = T;
    int stride_t = D_IN;  /* step along time axis in the input sequence */

    printf("Sequence [N=%d][T=%d][D_in=%d]: (T is dynamic runtime param)\n",
           N, seq_len, D_IN);

    /* Input sequence: N=1, T=3, D_in=2 → [[1,0],[0,1],[1,1]] */
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
     * Weights: small identity-like matrices (0.5 on diagonal), biases=0.
     * W shape [D_h][D_c] where D_c = D_in + D_h = 4
     * Flat: dh * D_C + dc
     *
     * For D_h=2, D_c=4:
     *   row0 = [0.5, 0,   0.5, 0  ]
     *   row1 = [0,   0.5, 0,   0.5]
     * This picks x[0]+h[0] for dh=0, x[1]+h[1] for dh=1.
     */
    float W_i[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,   /* row dh=0 */
        0.0f, 0.5f, 0.0f, 0.5f    /* row dh=1 */
    };
    float W_f[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.5f
    };
    float W_g[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.5f
    };
    float W_o[D_H * D_C] = {
        0.5f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.5f
    };

    /* biases = 0 for all gates */
    float b_i[D_H] = {0.0f, 0.0f};
    float b_f[D_H] = {0.0f, 0.0f};
    float b_g[D_H] = {0.0f, 0.0f};
    float b_o[D_H] = {0.0f, 0.0f};

    /* initial hidden and cell states = zeros */
    float h[N * D_H] = {0.0f, 0.0f};
    float c[N * D_H] = {0.0f, 0.0f};

    float h_new[N * D_H];
    float c_new[N * D_H];

    printf("\nUnrolling LSTM over T=%d steps:\n", seq_len);
    printf("(initial h=[0,0]  c=[0,0])\n\n");

    /* t: time step — LSTM unrolled over dynamic T */
    for (t = 0; t < seq_len; t++)
    {
        /* x_t is a slice of x_seq at time t: flat offset t*stride_t per sample */
        const float *x_t = x_seq + 0 * T * D_IN + t * stride_t; /* n=0 slice */

        lstm_step(x_t, h, c,
                  W_i, W_f, W_g, W_o,
                  b_i, b_f, b_g, b_o,
                  h_new, c_new,
                  D_IN, D_H, N);

        /* copy new states into h, c for next step */
        memcpy(h, h_new, N * D_H * sizeof(float));
        memcpy(c, c_new, N * D_H * sizeof(float));

        printf("  t=%d  h=[", t);
        for (d = 0; d < D_H; d++)
        {
            printf(" %7.4f", h[d]);
        }
        printf(" ]  c=[");
        for (d = 0; d < D_H; d++)
        {
            printf(" %7.4f", c[d]);
        }
        printf(" ]\n");
    }

    /* Check: h values are in (-1,1) — tanh bounded output */
    printf("\nCHECK: h values in (-1,1) from tanh bound:\n");
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
