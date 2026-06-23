/*
 * Operator  : Leaky ReLU
 * Formula   : f(x) = x > 0 ? x : alpha * x,  alpha = 0.1
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : N multiplies + N comparisons
 * Used in   : YOLOv3/v5 detection heads — avoids dying ReLU problem
 *             where neurons stuck at 0 gradient never recover
 * DEMO      : 6-element input; negatives get scaled by 0.1, positives pass through
 * Build     : gcc 02_leaky_relu.c -O2 -o 02_leaky_relu -lm
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

/* Apply Leaky ReLU: negatives are scaled by alpha instead of zeroed */
static void leaky_relu(const float *in, float *out, int N, float alpha)
{
    int i;
    for (i = 0; i < N; i++)
    {
        /* positive: identity; negative: scale by alpha (small leak, not zero) */
        out[i] = in[i] > 0.0f ? in[i] : alpha * in[i];
    }
}

int main(void)
{
    float v[]  = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f};
    int   N    = 6;
    float alpha = 0.1f;
    float out[6];

    leaky_relu(v, out, N, alpha);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: out[0] = -3 * 0.1 = -0.3 */
    float expected = -3.0f * alpha;   /* -0.3 */
    printf("CHECK out[0]=%.4f == -0.3000 : %s\n",
           out[0], (fabsf(out[0] - expected) < 1e-6f) ? "PASS" : "FAIL");

    return 0;
}
