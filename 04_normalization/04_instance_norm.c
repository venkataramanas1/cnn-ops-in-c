/*
 * Operator  : Instance Normalization
 * Normalizes over: each channel's H×W spatial pixels independently (per sample)
 *   — one mean+var per (sample, channel) pair
 * Models    : Style transfer (AdaIN uses IN statistics), CycleGAN,
 *             domain-adaptation layers in some VLA pipelines
 * DEMO      : C=2, H=3, W=3
 *   ch0: values around 1    (small magnitude)
 *   ch1: values around 100  (large magnitude)
 *   Both channels independently normalized → comparable outputs
 * Build     : gcc -O2 -o instance_norm 04_instance_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define C   2
#define H   3
#define W   3
#define EPS 1e-5f

/* Print a [C][H][W] tensor with a title */
static void show(const char *title, const float *m, int c_dim, int h_dim, int w_dim)
{
    int c, h, w;
    printf("%s\n", title);
    for (c = 0; c < c_dim; c++)
    {
        printf("  ch%d:\n", c);
        for (h = 0; h < h_dim; h++)
        {
            printf("    ");
            for (w = 0; w < w_dim; w++)
            {
                /* flat index: c * H*W + h * W + w */
                printf("%8.4f", m[c * h_dim * w_dim + h * w_dim + w]);
            }
            printf("\n");
        }
    }
}

/*
 * instance_norm: normalize each channel over its H×W pixels independently.
 *
 * InstanceNorm removes per-image style statistics — the stats of each channel
 * at each sample vanish, leaving only structure.
 *
 * in    : [C][H][W] input
 * out   : [C][H][W] output
 * gamma : [C] per-channel scale
 * beta  : [C] per-channel shift
 */
static void instance_norm(const float *in, float *out,
                          const float *gamma, const float *beta,
                          int c_dim, int h_dim, int w_dim)
{
    int c, h, w;
    int spatial = h_dim * w_dim;   /* number of pixels per channel */

    for (c = 0; c < c_dim; c++)   /* each channel normalized independently */
    {
        /* Step 1: compute mean over H×W pixels of this channel */
        float mean = 0.0f;
        for (h = 0; h < h_dim; h++)
        {
            for (w = 0; w < w_dim; w++)
            {
                mean += in[c * spatial + h * w_dim + w];   /* accumulate pixel */
            }
        }
        mean /= (float)spatial;    /* divide by spatial pixel count */

        /* Step 2: compute variance over H×W pixels of this channel */
        float var = 0.0f;
        for (h = 0; h < h_dim; h++)
        {
            for (w = 0; w < w_dim; w++)
            {
                float diff = in[c * spatial + h * w_dim + w] - mean;   /* deviation */
                var += diff * diff;   /* squared deviation */
            }
        }
        var /= (float)spatial;    /* average squared deviation = variance */

        /* Step 3: normalize each pixel and apply per-channel affine transform */
        float inv_std = 1.0f / sqrtf(var + EPS);   /* 1/std with epsilon guard */
        for (h = 0; h < h_dim; h++)
        {
            for (w = 0; w < w_dim; w++)
            {
                int idx = c * spatial + h * w_dim + w;        /* flat index */
                float xhat = (in[idx] - mean) * inv_std;      /* zero-mean unit-var */
                out[idx] = gamma[c] * xhat + beta[c];         /* affine transform */
            }
        }
    }
}

int main(void)
{
    /* ch0: values around 1 (small); ch1: values around 100 (large) */
    float input[C * H * W] = {
        /* ch0 */
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        /* ch1: 100x scale */
        100.0f, 200.0f, 300.0f,
        400.0f, 500.0f, 600.0f,
        700.0f, 800.0f, 900.0f
    };

    float output[C * H * W];

    /* identity affine */
    float gamma[C] = {1.0f, 1.0f};
    float beta[C]  = {0.0f, 0.0f};

    show("Input:", input, C, H, W);

    instance_norm(input, output, gamma, beta, C, H, W);

    show("Output (InstanceNorm):", output, C, H, W);

    /* Correctness: each channel of output should have mean≈0 and var≈1 */
    int c, i;
    int pass = 1;
    int spatial = H * W;

    for (c = 0; c < C; c++)
    {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (i = 0; i < spatial; i++)
        {
            float v = output[c * spatial + i];
            sum    += v;
            sum_sq += v * v;
        }
        float mean_out = sum / (float)spatial;
        float var_out  = sum_sq / (float)spatial - mean_out * mean_out;

        printf("\nch%d: mean=%.5f (expect ~0), var=%.5f (expect ~1)\n",
               c, mean_out, var_out);

        if (fabsf(mean_out) > 1e-3f || fabsf(var_out - 1.0f) > 0.05f)
        {
            pass = 0;
        }
    }

    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
