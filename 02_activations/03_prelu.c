/*
 * Operator  : PReLU (Parametric ReLU)
 * Formula   : f(x_i) = x_i > 0 ? x_i : alpha[i] * x_i  (per-element alpha)
 * Shape     : elementwise — input [N], alpha [N] -> output [N]
 * FLOPs     : N multiplies + N comparisons
 * Used in   : PReLU-Net (He et al. 2015), ArcFace face recognition
 *             Each channel learns its own negative slope during training
 * DEMO      : N=4, alternating alphas show per-element behaviour
 * Build     : gcc 03_prelu.c -O2 -o 03_prelu -lm
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

/* PReLU: each element has its own learned negative slope */
static void prelu(const float *in, const float *alpha, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        /* positive: pass through; negative: scale by this element's alpha */
        out[i] = in[i] > 0.0f ? in[i] : alpha[i] * in[i];
    }
}

int main(void)
{
    float x[]     = {-2.0f,  1.0f, -2.0f,  1.0f};
    float alpha[] = { 0.1f,  0.1f,  0.3f,  0.3f};
    int   N       = 4;
    float out[4];

    prelu(x, alpha, out, N);

    show("INPUT ", x,   N);
    show("ALPHA ", alpha, N);
    show("OUTPUT", out, N);

    /* Check: out[0] = -2 * 0.1 = -0.2 */
    printf("CHECK out[0]=%.4f == -0.2000 : %s\n",
           out[0], (fabsf(out[0] - (-0.2f)) < 1e-6f) ? "PASS" : "FAIL");

    /* Check: out[2] = -2 * 0.3 = -0.6 */
    printf("CHECK out[2]=%.4f == -0.6000 : %s\n",
           out[2], (fabsf(out[2] - (-0.6f)) < 1e-6f) ? "PASS" : "FAIL");

    return 0;
}
