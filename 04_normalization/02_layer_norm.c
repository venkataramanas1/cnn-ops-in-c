/*
 * Operator  : Layer Normalization (1-D token version)
 * Normalizes over: all D elements of a single token/feature vector
 *   (in image context: all C*H*W values per sample)
 * Models    : ViT, BERT, GPT, ALL VLA models (RT-2, OpenVLA, π0, Octo)
 *             — LayerNorm after every attention block and FFN block
 * DEMO      : vector of length D=8, values [1,2,3,4,5,6,7,8]
 *             output should have mean≈0 and std≈1
 * Build     : gcc -O2 -o layer_norm 02_layer_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define D   8
#define EPS 1e-5f

/*
 * show: print a 1-D vector (treated as [1][1][D] for uniform signature)
 * C/H/W unused beyond W=length for the 1D case; signature kept uniform.
 */
static void show(const char *title, const float *m, int C, int H, int W)
{
    (void)C; (void)H;   /* unused for 1-D demo */
    int i;
    printf("%s\n  [ ", title);
    for (i = 0; i < W; i++)
    {
        printf("%8.4f", m[i]);
    }
    printf(" ]\n");
}

/*
 * layer_norm: normalize a flat vector of length d.
 *
 * LayerNorm normalizes each token independently — essential for transformers;
 * unlike BN it does not depend on batch size.
 *
 * in    : [D] input vector
 * out   : [D] output vector
 * gamma : [D] per-element scale  (learned)
 * beta  : [D] per-element shift  (learned)
 */
static void layer_norm(const float *in, float *out,
                       const float *gamma, const float *beta, int d)
{
    int i;

    /* Step 1: compute mean over all D elements */
    float mean = 0.0f;
    for (i = 0; i < d; i++)
    {
        mean += in[i];   /* accumulate each element */
    }
    mean /= (float)d;    /* divide by dimension size */

    /* Step 2: compute variance over all D elements */
    float var = 0.0f;
    for (i = 0; i < d; i++)
    {
        float diff = in[i] - mean;   /* deviation from mean */
        var += diff * diff;           /* squared deviation */
    }
    var /= (float)d;    /* average squared deviation = variance */

    /* Step 3: normalize each element and apply affine transform */
    float inv_std = 1.0f / sqrtf(var + EPS);   /* 1/std with epsilon guard */
    for (i = 0; i < d; i++)
    {
        float xhat = (in[i] - mean) * inv_std;  /* zero-mean, unit-variance */
        out[i] = gamma[i] * xhat + beta[i];     /* affine: scale + shift */
    }
}

int main(void)
{
    float input[D]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[D];

    /* identity affine: gamma=1, beta=0 */
    float gamma[D] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float beta[D]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    show("Input:", input, 1, 1, D);

    layer_norm(input, output, gamma, beta, D);

    show("Output (LayerNorm):", output, 1, 1, D);

    /* Correctness: mean of output ≈ 0, var of output ≈ 1 */
    float mean_out = 0.0f;
    int i;
    for (i = 0; i < D; i++)
    {
        mean_out += output[i];
    }
    mean_out /= (float)D;

    float var_out = 0.0f;
    for (i = 0; i < D; i++)
    {
        float diff = output[i] - mean_out;
        var_out += diff * diff;
    }
    var_out /= (float)D;

    printf("\nCheck: mean=%.5f (expect ~0), var=%.5f (expect ~1)\n",
           mean_out, var_out);

    int pass = (fabsf(mean_out) < 1e-4f) && (fabsf(var_out - 1.0f) < 0.05f);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
