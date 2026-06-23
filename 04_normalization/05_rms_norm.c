/*
 * Operator  : RMS Normalization (Root Mean Square Norm)
 * Normalizes over: all D elements of a vector — scale only, no mean centering
 * Formula   : y[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
 * Models    : LLaMA, Mistral, RT-2 language decoder, OpenVLA
 *             — RMSNorm replaced LayerNorm in most modern LLM-based VLA models
 * DEMO      : vector [1,2,3,4,5,6,7,8], gamma=all 1.0
 *             output magnitudes normalized; mean is NOT forced to zero
 * Build     : gcc -O2 -o rms_norm 05_rms_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define D   8
#define EPS 1e-5f

/* Print a 1-D vector (C/H unused for 1-D demo; signature kept uniform) */
static void show(const char *title, const float *m, int C, int H, int W)
{
    (void)C; (void)H;
    int i;
    printf("%s\n  [ ", title);
    for (i = 0; i < W; i++)
    {
        printf("%8.4f", m[i]);
    }
    printf(" ]\n");
}

/*
 * rms_norm: normalize a flat vector by its RMS value.
 *
 * RMSNorm drops the mean-centering of LayerNorm — faster and empirically
 * as good; used in LLaMA/RT-2/OpenVLA language decoder.
 *
 * in    : [D] input vector
 * out   : [D] output vector
 * gamma : [D] per-element scale (learned)
 */
static void rms_norm(const float *in, float *out, const float *gamma, int d)
{
    int i;

    /* Step 1: compute mean of squares (no mean subtraction — that is the key
     * difference from LayerNorm; only the scale is normalized) */
    float mean_sq = 0.0f;
    for (i = 0; i < d; i++)
    {
        mean_sq += in[i] * in[i];   /* accumulate squared element */
    }
    mean_sq /= (float)d;            /* divide by dimension size */

    /* Step 2: compute RMS = sqrt(mean_sq + eps)
     * (there is no variance step — RMSNorm has no variance, only RMS) */
    float rms = sqrtf(mean_sq + EPS);   /* RMS with epsilon guard */
    float inv_rms = 1.0f / rms;         /* reciprocal for efficiency */

    /* Step 3: normalize and apply per-element scale (no bias in RMSNorm) */
    for (i = 0; i < d; i++)
    {
        out[i] = (in[i] * inv_rms) * gamma[i];   /* scale by 1/RMS then by gamma */
    }
}

int main(void)
{
    float input[D]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[D];

    /* identity scale: gamma=1 */
    float gamma[D] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    show("Input:", input, 1, 1, D);

    rms_norm(input, output, gamma, D);

    show("Output (RMSNorm):", output, 1, 1, D);

    /* Correctness: RMS of output should be ~1 */
    float sum_sq = 0.0f;
    int i;
    for (i = 0; i < D; i++)
    {
        sum_sq += output[i] * output[i];
    }
    float rms_out = sqrtf(sum_sq / (float)D);

    printf("\nCheck: RMS of output = %.5f (expect ~1.0)\n", rms_out);

    int pass = fabsf(rms_out - 1.0f) < 1e-4f;
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
