/*
 * Operator  : GELU (Gaussian Error Linear Unit)
 * Formula   : f(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 *             where sqrt(2/pi) = 0.7978845608
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : ~5 ops per element (mul, add, tanh, mul, mul)
 * Used in   : ViT, BERT, GPT, RT-2, OpenVLA, pi0
 *             GELU in every transformer FFN block — the dominant activation
 *             for large-scale vision-language-action models
 * DEMO      : 7-element sweep from -3 to 3 showing smooth gating behaviour
 * Build     : gcc 05_gelu.c -O2 -o 05_gelu -lm
 */

#include <math.h>
#include <stdio.h>

/* sqrt(2/pi) precomputed */
#define SQRT_2_OVER_PI 0.7978845608f

static void show(const char *title, const float *v, int N)
{
    printf("%s: ", title);
    int i;
    for (i = 0; i < N; i++)
    {
        printf("%.4f", v[i]);
        if (i < N - 1)
        {
            printf("  ");
        }
    }
    printf("\n");
}

/* GELU: gates input by its own cumulative distribution
 * near-0 values are attenuated, large magnitudes pass through.
 * Used in ALL VLA transformer FFN blocks (RT-2, OpenVLA, pi0). */
static void gelu(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        float x = in[i];
        /* cubic correction term: 0.044715 * x^3 */
        float x3 = 0.044715f * x * x * x;
        /* tanh argument: sqrt(2/pi) * (x + cubic) */
        float inner = SQRT_2_OVER_PI * (x + x3);
        /* gate: 0.5*(1 + tanh(inner)) is approx Phi(x), the Gaussian CDF */
        float gate = 0.5f * (1.0f + tanhf(inner));
        /* output: input scaled by its own gate */
        out[i] = x * gate;
    }
}

int main(void)
{
    float v[] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    int   N   = 7;
    float out[7];

    gelu(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: GELU(0) = 0 exactly */
    printf("CHECK out[3]=%.4f == 0.0000 : %s\n",
           out[3], (out[3] == 0.0f) ? "PASS" : "FAIL");

    return 0;
}
