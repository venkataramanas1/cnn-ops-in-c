/*
 * Operator  : ReLU (Rectified Linear Unit)
 * Formula   : f(x) = max(0, x)
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : N comparisons (no multiply)
 * Used in   : ResNet, MobileNet, YOLOv3 — default activation pre-2020
 * DEMO      : 6-element input with negatives, zero, and positives
 * Build     : gcc 01_relu.c -O2 -o 01_relu -lm
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

/* Apply ReLU elementwise: negative values are clamped to 0 */
static void relu(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        /* if x > 0 keep it; otherwise output 0 */
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

int main(void)
{
    float v[]   = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f};
    int   N     = 6;
    float out[6];

    relu(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: positive value passes through unchanged */
    printf("CHECK out[3]=%.4f == 1.0000 : %s\n",
           out[3], (out[3] == 1.0f) ? "PASS" : "FAIL");

    /* Check: negative value is clamped to 0 */
    printf("CHECK out[0]=%.4f == 0.0000 : %s\n",
           out[0], (out[0] == 0.0f) ? "PASS" : "FAIL");

    return 0;
}
