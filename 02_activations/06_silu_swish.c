/*
 * Operator  : SiLU / Swish
 * Formula   : f(x) = x / (1 + exp(-x))  =  x * sigmoid(x)
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : 2N ops (1 expf + 1 divide per element)
 * Used in   : EfficientNet, MobileNetV3, LLaMA, RT-2 action decoder
 *             Outperforms ReLU on deep nets; smooth and non-monotone
 * DEMO      : 7-element sweep showing the slight negative dip near x=-1.28
 * Build     : gcc 06_silu_swish.c -O2 -o 06_silu_swish -lm
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

/* SiLU = x * sigmoid(x): the sigmoid acts as a soft gate on its own input.
 * Slight negative dip around x=-1.28 is unique to SiLU. */
static void silu(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        float x = in[i];
        /* sigmoid(x) = 1 / (1 + exp(-x)) */
        float sig = 1.0f / (1.0f + expf(-x));
        /* SiLU gates x by its own sigmoid */
        out[i] = x * sig;
    }
}

int main(void)
{
    float v[] = {-4.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f};
    int   N   = 7;
    float out[7];

    silu(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0 exactly */
    printf("CHECK out[3]=%.4f == 0.0000 : %s\n",
           out[3], (out[3] == 0.0f) ? "PASS" : "FAIL");

    return 0;
}
