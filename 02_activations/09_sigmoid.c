/*
 * Operator  : Sigmoid
 * Formula   : f(x) = 1 / (1 + exp(-x))
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : 2N ops per element (expf + divide)
 * Used in   : SE block channel gates, LSTM gates, binary classifiers,
 *             VLA action confidence heads
 * DEMO      : 7-element sweep from -6 to +6 showing saturation at 0 and 1
 * Build     : gcc 09_sigmoid.c -O2 -o 09_sigmoid -lm
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

/* sigmoid output in (0,1) — saturates near 0 and 1 for large magnitudes.
 * Used as a gate (SE blocks, LSTM) or binary probability. */
static void sigmoid(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        /* 1 / (1 + e^-x): always in (0, 1), smooth, differentiable */
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

int main(void)
{
    float v[] = {-6.0f, -3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 6.0f};
    int   N   = 7;
    float out[7];

    sigmoid(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: sigmoid(0) = 0.5 exactly */
    printf("CHECK out[3]=%.6f ~= 0.500000 : %s\n",
           out[3], (fabsf(out[3] - 0.5f) < 1e-6f) ? "PASS" : "FAIL");

    return 0;
}
