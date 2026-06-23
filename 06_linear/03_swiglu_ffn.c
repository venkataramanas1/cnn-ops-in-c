/*
 * 03_swiglu_ffn.c
 *
 * Op: SwiGLU feed-forward block — the FFN used in LLaMA / RT-2 / OpenVLA.
 *
 *   Given input x [N][T][D]:
 *     gate [N][T][D_ff]  = x @ W_gate    (linear projection)
 *     up   [N][T][D_ff]  = x @ W_up      (linear projection)
 *     h    [N][T][D_ff]  = silu(gate) * up   (elementwise gate × up values)
 *     out  [N][T][D]     = h @ W_down    (project back to model dim)
 *
 *   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * 4D note: all intermediate tensors are [N][T][...];
 *          the FFN is applied independently per token (n, t).
 *
 * Comment: "SwiGLU = SiLU(gate) * up: the gate vector controls HOW MUCH of
 *  the up-projected values pass through. Two separate linear projections
 *  (gate and up) give the model more expressive power than a single
 *  projection + activation. Used in LLaMA and every LLM-based VLA
 *  (RT-2, OpenVLA)."
 *
 * WHERE IN MODELS:
 *   - LLaMA (all sizes), Mistral, Gemma: SwiGLU replaced vanilla FFN (GELU).
 *   - RT-2 / OpenVLA language decoder: same SwiGLU FFN in every transformer
 *     block that processes the joint vision+language token sequence.
 *   - Vanilla transformers (BERT/ViT): used GELU FFN instead — SwiGLU is the
 *     modern replacement.
 *
 * BUILD:
 *   gcc -O2 -o 03_swiglu_ffn 03_swiglu_ffn.c -lm && ./03_swiglu_ffn
 */

#include <stdio.h>
#include <math.h>

/* Static shapes for this demo */
#define N     1
#define T     2
#define D     4    /* model dimension (in and out of the FFN) */
#define D_FF  8    /* feed-forward expansion dimension        */

/* ------------------------------------------------------------------ */
/* silu: SiLU / Swish activation                                        */
/*   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))                     */
/* ------------------------------------------------------------------ */
static float silu(float x)
{
    return x / (1.0f + expf(-x)); /* x * sigmoid(x) */
}

/* ------------------------------------------------------------------ */
/* linear_proj                                                          */
/*   in  [N][T][d_in]  →  out [N][T][d_out]                           */
/*   out[n][t][do] = sum_{di} in[n][t][di] * W[do][di]  (no bias)     */
/*                                                                      */
/* Dynamic strides (computed at runtime, no compile-time constants):    */
/*   int stride_n_in  = t_dim * d_in;   step to next batch in input   */
/*   int stride_t_in  = d_in;           step to next token in input    */
/*   int stride_n_out = t_dim * d_out;  step to next batch in output   */
/*   int stride_t_out = d_out;          step to next token in output   */
/*   int stride_w_row = d_in;           step to next row in W          */
/* ------------------------------------------------------------------ */
static void linear_proj(const float *in,  /* [N][T][d_in]  */
                        const float *W,   /* [d_out][d_in] */
                        float       *out, /* [N][T][d_out] */
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
                float acc = 0.0f;
                for (di = 0; di < d_in; di++)
                {
                    /* in[n][t][di]:  n*stride_n_in  + t*stride_t_in  + di  */
                    /* W[dout][di]:   dout*stride_w_row + di                 */
                    acc += in[n * stride_n_in + t * stride_t_in + di]
                         * W[dout * stride_w_row + di];
                }
                /* out[n][t][dout]: n*stride_n_out + t*stride_t_out + dout */
                out[n * stride_n_out + t * stride_t_out + dout] = acc;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* elementwise_swiglu                                                   */
/*   h[n][t][k] = silu(gate[n][t][k]) * up[n][t][k]                   */
/*   all tensors [N][T][D_ff], same layout                             */
/* ------------------------------------------------------------------ */
static void elementwise_swiglu(const float *gate, /* [N][T][D_ff] */
                                const float *up,   /* [N][T][D_ff] */
                                float       *h,    /* [N][T][D_ff] */
                                int n_dim, int t_dim, int dff_dim)
{
    /* dynamic strides: computed at runtime, no compile-time constants */
    int stride_n = t_dim * dff_dim; /* step to next batch item */
    int stride_t = dff_dim;         /* step to next token      */

    int n, t, k;
    for (n = 0; n < n_dim; n++)
    {
        for (t = 0; t < t_dim; t++)
        {
            for (k = 0; k < dff_dim; k++)
            {
                /* gate[n][t][k]: n*stride_n + t*stride_t + k */
                float g = gate[n * stride_n + t * stride_t + k];
                float u = up[n * stride_n + t * stride_t + k];
                /* silu(gate) * up: gate controls how much of up passes */
                h[n * stride_n + t * stride_t + k] = silu(g) * u;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* show helpers                                                         */
/* ------------------------------------------------------------------ */

static void show_ntd(const char *label, const float *buf,
                     int n_dim, int t_dim, int d_dim)
{
    int n, t, d;
    printf("  %s [N=%d][T=%d][D=%d]:\n", label, n_dim, t_dim, d_dim);
    for (n = 0; n < n_dim; n++)
    {
        for (t = 0; t < t_dim; t++)
        {
            printf("  n=%d t=%d: [", n, t);
            for (d = 0; d < d_dim; d++)
            {
                /* flat: n*T*D + t*D + d */
                printf(" %6.3f", buf[n * t_dim * d_dim + t * d_dim + d]);
            }
            printf(" ]\n");
        }
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    int i;

    /*
     * Input x [N=1][T=2][D=4]: ramp 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
     * Moderate positive values so silu gating effect is clearly visible.
     */
    float x[N * T * D];
    for (i = 0; i < N * T * D; i++)
    {
        x[i] = (float)(i + 1) * 0.5f; /* 0.5, 1.0, 1.5, ... */
    }

    /*
     * W_gate [D_ff=8][D=4]: all-0.5 weight matrix.
     * Each gate output = 0.5 * sum of input features.
     * Produces moderate gate values that land in the nonlinear silu region.
     * flat: dff*D + di
     */
    float W_gate[D_FF * D];
    for (i = 0; i < D_FF * D; i++)
    {
        W_gate[i] = 0.5f;
    }

    /*
     * W_up [D_ff=8][D=4]: identity-like — each up output selects one feature.
     * Row k selects input feature (k % D).
     * This makes the up-projection values easy to trace back to the input.
     * flat: dff*D + di
     */
    float W_up[D_FF * D];
    for (i = 0; i < D_FF * D; i++)
    {
        W_up[i] = 0.0f;
    }
    {
        int k;
        for (k = 0; k < D_FF; k++)
        {
            /* W_up[k][k % D] = 1  →  up[k] = x[k % D] */
            W_up[k * D + (k % D)] = 1.0f; /* flat: k*D + (k%D) */
        }
    }

    /*
     * W_down [D=4][D_ff=8]: average the 8 hidden values back into 4 dims.
     * Row d of W_down sums pairs: W_down[d][k] = 0.5 when k/2 == d.
     * flat: d*D_ff + k
     */
    float W_down[D * D_FF];
    for (i = 0; i < D * D_FF; i++)
    {
        W_down[i] = 0.0f;
    }
    {
        int d;
        for (d = 0; d < D; d++)
        {
            /* each output dim averages two adjacent hidden units */
            W_down[d * D_FF + d * 2]     = 0.5f; /* flat: d*D_ff + d*2     */
            W_down[d * D_FF + d * 2 + 1] = 0.5f; /* flat: d*D_ff + d*2 + 1 */
        }
    }

    /* Intermediate buffers */
    float gate[N * T * D_FF];
    float up[N * T * D_FF];
    float h[N * T * D_FF];
    float out[N * T * D];

    printf("=== 03_swiglu_ffn ===\n\n");

    printf("INPUT x:\n");
    show_ntd("x", x, N, T, D);

    /* Step 1: gate = x @ W_gate */
    linear_proj(x, W_gate, gate, N, T, D, D_FF);
    printf("\nSTEP 1 — gate = x @ W_gate:\n");
    show_ntd("gate", gate, N, T, D_FF);

    /* Step 2: up = x @ W_up */
    linear_proj(x, W_up, up, N, T, D, D_FF);
    printf("\nSTEP 2 — up = x @ W_up:\n");
    show_ntd("up", up, N, T, D_FF);

    /* Step 3: h = silu(gate) * up */
    elementwise_swiglu(gate, up, h, N, T, D_FF);
    printf("\nSTEP 3 — h = silu(gate) * up  [gating controls up-values]:\n");
    show_ntd("h = silu(gate)*up", h, N, T, D_FF);

    /* Step 4: out = h @ W_down */
    linear_proj(h, W_down, out, N, T, D_FF, D);
    printf("\nSTEP 4 — out = h @ W_down:\n");
    show_ntd("out", out, N, T, D);

    /*
     * CHECK: output has been gated (values differ from plain linear).
     * A plain linear x @ (W_gate * W_down) would give different values
     * because silu(gate) modulates the up values non-linearly.
     *
     * Concretely: gate[n=0][t=0] = 0.5*(0.5+1.0+1.5+2.0) = 2.5 (all rows same)
     * silu(2.5) = 2.5 / (1 + exp(-2.5)) ≈ 2.5 * 0.924 ≈ 2.31
     * up[0][0][0] = x[0][0][0 % 4] = x[0][0][0] = 0.5
     * h[0][0][0] = silu(2.5) * 0.5 ≈ 1.155  (gate dampens for small up values)
     */
    printf("\nCHECK: out[0][0][0] = %.4f\n",
           out[0 * T * D + 0 * D + 0]); /* flat: 0*T*D + 0*D + 0 */
    printf("  gate[0][0][0] = %.4f,  silu(gate[0][0][0]) = %.4f\n",
           gate[0 * T * D_FF + 0 * D_FF + 0], /* flat: 0*T*D_ff + 0*D_ff + 0 */
           silu(gate[0 * T * D_FF + 0 * D_FF + 0]));
    printf("  up[0][0][0]   = %.4f\n",
           up[0 * T * D_FF + 0 * D_FF + 0]); /* flat: 0*T*D_ff + 0*D_ff + 0 */
    printf("  h[0][0][0]    = silu(gate)*up = %.4f\n",
           h[0 * T * D_FF + 0 * D_FF + 0]); /* flat: 0*T*D_ff + 0*D_ff + 0 */
    printf("  (values are gated — NOT equal to plain sum-of-inputs)\n");

    return 0;
}
