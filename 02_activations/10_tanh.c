/*
 * Operator  : Tanh
 * Formula   : f(x) = tanh(x)
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : ~4N ops per element (two expf + add + divide, or hardware tanhf)
 * Used in   : LSTM/GRU cell gates, some VLA recurrent action decoders
 *             Output in [-1, 1]; steeper slope near 0 than sigmoid
 * DEMO      : 7-element sweep showing bounded output and steep centre
 * Build     : gcc 10_tanh.c -O2 -o 10_tanh -lm
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

/* tanh output in (-1,1) — steeper than sigmoid around 0.
 * Used for LSTM cell state; output stays bounded which helps gradient flow. */
static void apply_tanh(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        /* tanhf: zero-centred unlike sigmoid, saturates at +-1 */
        out[i] = tanhf(in[i]);
    }
}

int main(void)
{
    float v[] = {-4.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f};
    int   N   = 7;
    float out[7];

    apply_tanh(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: tanh(0) = 0 exactly */
    printf("CHECK out[3]=%.4f == 0.0000 : %s\n",
           out[3], (out[3] == 0.0f) ? "PASS" : "FAIL");

    return 0;
}
