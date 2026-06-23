/*
 * Operator  : L2 Normalization
 * Normalizes over: all elements of a vector to produce unit L2 norm
 * Formula   : y[i] = x[i] / (sqrt(sum(x[j]^2)) + eps)
 * Models    : Face recognition embedding heads (ArcFace, CosFace),
 *             cross-modal matching in VLA (vision-language cosine similarity)
 * DEMO      : vector [3, 4] → output [0.6, 0.8]
 *             classic 3-4-5 right triangle; L2 norm of [3,4] = 5
 * Build     : gcc -O2 -o l2_norm 06_l2_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define D   2
#define EPS 1e-8f   /* small epsilon; L2 norm is not a variance, needs tiny guard */

/* Print a 1-D vector (C/H unused; signature kept uniform) */
static void show(const char *title, const float *m, int C, int H, int W)
{
    (void)C; (void)H;
    int i;
    printf("%s\n  [ ", title);
    for (i = 0; i < W; i++)
    {
        printf("%8.6f", m[i]);
    }
    printf(" ]\n");
}

/*
 * l2_norm: project a vector onto the unit hypersphere.
 *
 * L2 norm maps the feature to a unit hypersphere — dot product becomes
 * cosine similarity. Used in embedding heads for metric learning.
 *
 * in  : [d] input vector
 * out : [d] output vector (unit L2 norm)
 */
static void l2_norm(const float *in, float *out, int d)
{
    int i;

    /* Step 1: compute sum of squares (L2 norm squared) */
    float sum_sq = 0.0f;
    for (i = 0; i < d; i++)
    {
        sum_sq += in[i] * in[i];   /* accumulate x[i]^2 */
    }

    /* Step 2: compute L2 norm (with epsilon guard against zero division) */
    float l2 = sqrtf(sum_sq) + EPS;   /* L2 = sqrt(sum of squares) + epsilon */

    /* Step 3: divide each element by the L2 norm (no affine — L2 norm has no gamma/beta) */
    for (i = 0; i < d; i++)
    {
        out[i] = in[i] / l2;   /* project onto unit sphere */
    }
}

int main(void)
{
    /* Classic 3-4-5 triangle: L2([3,4]) = 5 → [3/5, 4/5] = [0.6, 0.8] */
    float input[D]  = {3.0f, 4.0f};
    float output[D];

    show("Input:", input, 1, 1, D);

    l2_norm(input, output, D);

    show("Output (L2Norm):", output, 1, 1, D);

    /* Correctness: output[0] ≈ 0.6, output[1] ≈ 0.8, and L2 norm of output ≈ 1 */
    float l2_out = sqrtf(output[0] * output[0] + output[1] * output[1]);

    printf("\nCheck: output[0]=%.6f (expect 0.600000), output[1]=%.6f (expect 0.800000)\n",
           output[0], output[1]);
    printf("Check: L2 norm of output = %.6f (expect ~1.0)\n", l2_out);

    int pass = (fabsf(output[0] - 0.6f) < 1e-5f) &&
               (fabsf(output[1] - 0.8f) < 1e-5f) &&
               (fabsf(l2_out - 1.0f)    < 1e-5f);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
