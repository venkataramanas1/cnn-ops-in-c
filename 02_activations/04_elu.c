/*
 * Operator  : ELU (Exponential Linear Unit)
 * Formula   : f(x) = x >= 0 ? x : alpha * (exp(x) - 1),  alpha = 1.0
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : N comparisons + ~N/2 expf calls (only for negatives)
 * Used in   : ELU-Net — smooth negative region unlike ReLU's hard zero
 *             Self-normalising neural nets (SELU variant)
 * DEMO      : 5-element input; negative region shows smooth saturation toward -1
 * Build     : gcc 04_elu.c -O2 -o 04_elu -lm
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

/* ELU: smooth exponential curve for negatives; identity for positives */
static void elu(const float *in, float *out, int N, float alpha)
{
    int i;
    for (i = 0; i < N; i++)
    {
        if (in[i] >= 0.0f)
        {
            /* non-negative: identity */
            out[i] = in[i];
        }
        else
        {
            /* negative region: exp(x)-1 approaches -1 as x->-inf, smooth saturation */
            out[i] = alpha * (expf(in[i]) - 1.0f);
        }
    }
}

int main(void)
{
    float v[]  = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    int   N    = 5;
    float alpha = 1.0f;
    float out[5];

    elu(v, out, N, alpha);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: ELU(0) = 0 exactly — continuous at origin */
    printf("CHECK out[2]=%.4f == 0.0000 : %s\n",
           out[2], (out[2] == 0.0f) ? "PASS" : "FAIL");

    return 0;
}
