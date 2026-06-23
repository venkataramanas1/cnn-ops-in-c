/*
 * Operator  : HardSwish
 * Formula   : f(x) = 0          if x <= -3
 *                    x           if x >=  3
 *                    x*(x+3)/6   otherwise
 *             (equivalent to x * ReLU6(x+3) / 6)
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : N multiplies + N comparisons (no expf)
 * Used in   : MobileNetV3, EfficientNet-Lite — integer-friendly SiLU
 *             approximation for edge NPUs that lack fast exp units
 * DEMO      : 7-element sweep including the clamped regions and the midpoint
 * Build     : gcc 08_hard_swish.c -O2 -o 08_hard_swish -lm
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

/* HardSwish: piecewise linear approx of SiLU — no exp(), suitable for
 * fixed-point integer NPUs that lack fast exponential units */
static void hard_swish(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        float x = in[i];
        if (x <= -3.0f)
        {
            /* clamp to 0 below -3 */
            out[i] = 0.0f;
        }
        else if (x >= 3.0f)
        {
            /* identity above +3 */
            out[i] = x;
        }
        else
        {
            /* linear ramp: x*(x+3)/6 matches SiLU in middle region */
            out[i] = x * (x + 3.0f) / 6.0f;
        }
    }
}

int main(void)
{
    float v[] = {-4.0f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.0f};
    int   N   = 7;
    float out[7];

    hard_swish(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: clamped to 0 below -3 */
    printf("CHECK out[0]=%.4f == 0.0000 : %s\n",
           out[0], (out[0] == 0.0f) ? "PASS" : "FAIL");

    /* Check: identity above +3 */
    printf("CHECK out[6]=%.4f == 4.0000 : %s\n",
           out[6], (out[6] == 4.0f) ? "PASS" : "FAIL");

    return 0;
}
