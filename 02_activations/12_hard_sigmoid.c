/*
 * Operator  : HardSigmoid
 * Formula   : f(x) = clamp((x + 3) / 6,  0, 1)
 * Shape     : elementwise — input [N] -> output [N]
 * FLOPs     : 2N ops (add + divide) + N clamps — zero expf calls
 * Used in   : MobileNetV3 SE blocks on edge hardware — exp-free sigmoid
 *             replacement for fixed-point integer NPUs
 * DEMO      : 7-element sweep including clamped extremes and midpoint
 * Build     : gcc 12_hard_sigmoid.c -O2 -o 12_hard_sigmoid -lm
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

/* HardSigmoid: piecewise linear sigmoid with no exp() —
 * output in [0,1], safe for fixed-point integer NPUs */
static void hard_sigmoid(const float *in, float *out, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        /* linear ramp: (x+3)/6 maps [-3,3] -> [0,1] */
        float y = (in[i] + 3.0f) / 6.0f;
        /* clamp below 0 */
        if (y < 0.0f)
        {
            y = 0.0f;
        }
        /* clamp above 1 */
        if (y > 1.0f)
        {
            y = 1.0f;
        }
        out[i] = y;
    }
}

int main(void)
{
    float v[] = {-4.0f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.0f};
    int   N   = 7;
    float out[7];

    hard_sigmoid(v, out, N);

    show("INPUT ", v,   N);
    show("OUTPUT", out, N);

    /* Check: clamped to 0 below -3 */
    printf("CHECK out[0]=%.4f == 0.0000 : %s\n",
           out[0], (out[0] == 0.0f) ? "PASS" : "FAIL");

    /* Check: clamped to 1 above +3 */
    printf("CHECK out[6]=%.4f == 1.0000 : %s\n",
           out[6], (out[6] == 1.0f) ? "PASS" : "FAIL");

    /* Check: midpoint at x=0 gives 0.5 */
    printf("CHECK out[3]=%.4f == 0.5000 : %s\n",
           out[3], (out[3] == 0.5f) ? "PASS" : "FAIL");

    return 0;
}
