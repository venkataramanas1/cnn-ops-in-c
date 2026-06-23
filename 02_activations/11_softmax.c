/*
 * Operator  : Softmax (numerically stable)
 * Formula   : p[i] = exp(x[i] - max_x) / sum_exp
 * Shape     : input [N] -> output [N], outputs sum to 1.0
 * FLOPs     : N + N + N = 3N  (max scan + exp pass + normalise)
 * Used in   : Attention score normalisation in ViT/VLA MHSA,
 *             classification output heads (final layer probabilities)
 * DEMO      : 5-element logit vector; output is a probability distribution
 * Build     : gcc 11_softmax.c -O2 -o 11_softmax -lm
 */

#include <math.h>
#include <stdio.h>

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

/* Numerically stable softmax:
 * subtract max before exp — mathematically identical but prevents overflow
 * when logits are large (e.g. 100 vs 101 instead of raw exp(100)) */
static void softmax(const float *in, float *out, int N)
{
    int i;

    /* pass 1: find the maximum value to subtract for numerical stability */
    float max_x = in[0];
    for (i = 1; i < N; i++)
    {
        if (in[i] > max_x)
        {
            max_x = in[i];
        }
    }

    /* pass 2: compute shifted exponentials and accumulate sum */
    float sum_exp = 0.0f;
    for (i = 0; i < N; i++)
    {
        /* shift by max makes the largest exp = exp(0) = 1, rest < 1 */
        out[i] = expf(in[i] - max_x);
        sum_exp += out[i];
    }

    /* pass 3: normalise so all outputs sum to 1 */
    for (i = 0; i < N; i++)
    {
        out[i] /= sum_exp;
    }
}

int main(void)
{
    float v[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f};
    int   N   = 5;
    float out[5];

    softmax(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: outputs sum to 1.0 within floating-point tolerance */
    float sum = 0.0f;
    int i;
    for (i = 0; i < N; i++)
    {
        sum += out[i];
    }
    printf("CHECK sum=%.6f ~= 1.000000 : %s\n",
           sum, (fabsf(sum - 1.0f) < 1e-5f) ? "PASS" : "FAIL");

    return 0;
}
