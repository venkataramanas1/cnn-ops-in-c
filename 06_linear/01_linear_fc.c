/*
 * 01_linear_fc.c
 *
 * Op: Fully-connected / linear layer over a 3-D token tensor.
 *     Input  [N][T][D_in]  →  Output [N][T][D_out]
 *     y[n][t][do] = bias[do] + sum_{di} x[n][t][di] * W[do][di]
 *
 * Weight layout : [D_out][D_in],   flat: do * D_in + di
 * Input  layout : [N][T][D_in],    flat: n * T * D_in  + t * D_in  + di
 * Output layout : [N][T][D_out],   flat: n * T * D_out + t * D_out + do
 *
 * Comment: "Linear is the inner op of every transformer FFN and projection:
 *  applied independently to each (n,t) token, so the T dimension is just a
 *  batch loop."
 *
 * WHERE IN MODELS:
 *   - Every transformer FFN: two linear layers around an activation.
 *   - ViT/VLA projection layers, query/key/value projections, output heads.
 *   - Applied per-token — the (N,T) loops simply tile the same D_in→D_out map.
 *
 * Build:
 *   gcc -O2 -o 01_linear_fc 01_linear_fc.c -lm && ./01_linear_fc
 */

#include <stdio.h>
#include <math.h>

/* Static shapes for this demo */
#define N     2
#define T     3
#define D_IN  4
#define D_OUT 2

/* ------------------------------------------------------------------ */
/* show helpers                                                         */
/* ------------------------------------------------------------------ */

static void show_input(const float *x)
{
    int n, t, di;
    printf("  [N=%d][T=%d][D=%d]:\n", N, T, D_IN);
    for (n = 0; n < N; n++)
    {
        for (t = 0; t < T; t++)
        {
            printf("  n=%d t=%d: [", n, t);
            for (di = 0; di < D_IN; di++)
            {
                /* flat: n*T*D_in + t*D_in + di */
                printf(" %5.1f", x[n * T * D_IN + t * D_IN + di]);
            }
            printf(" ]\n");
        }
    }
}

static void show_output(const float *y)
{
    int n, t, dout;
    printf("  [N=%d][T=%d][D=%d]:\n", N, T, D_OUT);
    for (n = 0; n < N; n++)
    {
        for (t = 0; t < T; t++)
        {
            printf("  n=%d t=%d: [", n, t);
            for (dout = 0; dout < D_OUT; dout++)
            {
                /* flat: n*T*D_out + t*D_out + do */
                printf(" %5.1f", y[n * T * D_OUT + t * D_OUT + dout]);
            }
            printf(" ]\n");
        }
    }
}

/* ------------------------------------------------------------------ */
/* linear_fc                                                            */
/*                                                                      */
/* Dynamic strides (computed at runtime, no compile-time constants):    */
/*   int stride_n_in  = T * D_in;    step to next batch item in input  */
/*   int stride_t_in  = D_in;        step to next token in input       */
/*   int stride_n_out = T * D_out;   step to next batch item in output */
/*   int stride_t_out = D_out;       step to next token in output      */
/*   int stride_w_row = D_in;        step to next output-dim row in W  */
/*                                                                      */
/* With runtime shapes, replace every constant macro with those vars.   */
/* ------------------------------------------------------------------ */
static void linear_fc(const float *x,        /* [N][T][D_in]  */
                      const float *W,        /* [D_out][D_in] */
                      const float *bias,     /* [D_out]       */
                      float       *y,        /* [N][T][D_out] */
                      int n_dim, int t_dim, int d_in, int d_out)
{
    /* dynamic strides: computed at runtime, no compile-time constants */
    int stride_n_in  = t_dim * d_in;   /* step to next batch item in input  */
    int stride_t_in  = d_in;           /* step to next token in input       */
    int stride_n_out = t_dim * d_out;  /* step to next batch item in output */
    int stride_t_out = d_out;          /* step to next token in output      */
    int stride_w_row = d_in;           /* step to next output-dim row in W  */

    int n, t, dout, di;
    for (n = 0; n < n_dim; n++)
    {
        for (t = 0; t < t_dim; t++)
        {
            for (dout = 0; dout < d_out; dout++)
            {
                float acc = bias[dout]; /* start accumulator from bias */
                for (di = 0; di < d_in; di++)
                {
                    /* x[n][t][di]:    n*stride_n_in  + t*stride_t_in  + di */
                    /* W[dout][di]:    dout*stride_w_row + di               */
                    acc += x[n * stride_n_in + t * stride_t_in + di]
                         * W[dout * stride_w_row + di];
                }
                /* y[n][t][dout]:  n*stride_n_out + t*stride_t_out + dout */
                y[n * stride_n_out + t * stride_t_out + dout] = acc;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    int i;

    /* Input: ramp 1..24 reshaped to [N=2][T=3][D_in=4] */
    float x[N * T * D_IN];
    for (i = 0; i < N * T * D_IN; i++)
    {
        x[i] = (float)(i + 1); /* values 1 … 24 */
    }

    /*
     * W [D_out=2][D_in=4]:
     *   row 0 = [1, 0, 0, 0]  → selects first  feature (di=0)
     *   row 1 = [0, 1, 0, 0]  → selects second feature (di=1)
     * Flat: do*D_in + di
     */
    float W[D_OUT * D_IN] = {
        1.0f, 0.0f, 0.0f, 0.0f,   /* row 0: pick di=0 */
        0.0f, 1.0f, 0.0f, 0.0f    /* row 1: pick di=1 */
    };

    float bias[D_OUT] = {0.0f, 0.0f};

    float y[N * T * D_OUT];

    printf("=== 01_linear_fc ===\n\n");

    printf("INPUT x (ramp 1..24) [N=%d][T=%d][D_in=%d]:\n", N, T, D_IN);
    show_input(x);

    printf("\nWEIGHT W [D_out=%d][D_in=%d]"
           " (identity-like: row0 selects di=0, row1 selects di=1):\n",
           D_OUT, D_IN);
    {
        int dout, di;
        for (dout = 0; dout < D_OUT; dout++)
        {
            printf("  row %d: [", dout);
            for (di = 0; di < D_IN; di++)
            {
                /* W[dout][di]: dout*D_in + di */
                printf(" %.0f", W[dout * D_IN + di]);
            }
            printf(" ]\n");
        }
    }

    printf("\nbias: [ %.1f  %.1f ]\n", bias[0], bias[1]);

    linear_fc(x, W, bias, y, N, T, D_IN, D_OUT);

    printf("\nOUTPUT y [N=%d][T=%d][D_out=%d]:\n", N, T, D_OUT);
    show_output(y);

    /*
     * Check: out[0][0][0] should equal in[0][0][0] = 1.0
     * Because W[0][di] = [1,0,0,0], so y = bias[0] + x[0][0][0]*1 + rest*0 = 1.0
     */
    float expected = x[0 * T * D_IN + 0 * D_IN + 0]; /* x[0][0][0] = 1.0 */
    float got      = y[0 * T * D_OUT + 0 * D_OUT + 0]; /* y[0][0][0] */
    printf("\nCHECK: out[0][0][0] = %.1f  (expect %.1f from x[0][0][0])\n",
           got, expected);

    return 0;
}
