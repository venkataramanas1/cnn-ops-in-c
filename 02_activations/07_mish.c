/*
 * Operator  : Mish
 * Formula   : f(x) = x * tanh(softplus(x))  where softplus(x) = log(1 + exp(x))
 *             equivalently: f(x) = x * tanh(log1p(exp(x)))
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : ~3N ops per element (expf, log1p, tanh, mul)
 * Used in   : YOLOv4, Mish-Net — smoother tail than SiLU, unbounded above
 * DEMO      : 7-element sweep showing smooth non-monotone behaviour
 * Build     : gcc 07_mish.c -O2 -o 07_mish -lm
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

/* Mish = x * tanh(softplus(x)).
 * Use log1p(exp(x)) for softplus to avoid overflow for large x. */
static void mish(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        float x = in[i];
        /* softplus(x) = log(1 + e^x); log1p avoids cancellation near 0 */
        float sp = log1pf(expf(x));
        /* tanh of softplus is a smooth gate in (-1, 1) */
        float th = tanhf(sp);
        /* scale input by tanh gate */
        out[i] = x * th;
    }
}

int main(void)
{
    float v[] = {-4.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f};
    int   N   = 7;
    float out[7];

    mish(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: Mish(0) = 0 * tanh(log(2)) = 0 exactly */
    printf("CHECK out[3]=%.4f == 0.0000 : %s\n",
           out[3], (out[3] == 0.0f) ? "PASS" : "FAIL");

    return 0;
}
