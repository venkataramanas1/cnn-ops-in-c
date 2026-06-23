/*
 * Operator  : Batch Normalization (inference-time, per-channel)
 * Normalizes over: each channel's H×W spatial pixels independently
 *   (at training time, N samples form the batch; at inference, running stats are frozen)
 * Models    : ResNet, MobileNet, EfficientNet — BN after every conv during training
 * DEMO      : C=2, H=3, W=3
 *   ch0: all 1.0 → zero variance → output all 0.0
 *   ch1: [1..9]  → mean=5, clear normalization visible
 * Build     : gcc -O2 -o batch_norm 01_batch_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define C 2
#define H 3
#define W 3
#define EPS 1e-5f

/* Print a [C][H][W] tensor with a title */
static void show(const char *title, const float *m, int c_dim, int h_dim, int w_dim)
{
    printf("%s\n", title);
    int c, h, w;
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
 * batch_norm: normalize each channel over its H×W spatial pixels.
 *
 * At inference, mean and var are frozen running statistics from training —
 * here we compute them from the channel's spatial pixels as a stand-in.
 *
 * in      : [C][H][W] input feature map
 * out     : [C][H][W] output feature map
 * gamma   : [C]  scale  (learned affine parameter)
 * beta    : [C]  shift  (learned affine parameter)
 */
static void batch_norm(const float *in, float *out,
                       const float *gamma, const float *beta,
                       int c_dim, int h_dim, int w_dim)
{
    int c, h, w;
    int spatial = h_dim * w_dim;   /* number of pixels per channel */

    for (c = 0; c < c_dim; c++)
    {
        /* Step 1: compute mean over all H×W pixels in this channel */
        float mean = 0.0f;
        for (h = 0; h < h_dim; h++)
        {
            for (w = 0; w < w_dim; w++)
            {
                mean += in[c * spatial + h * w_dim + w];   /* accumulate pixel */
            }
        }
        mean /= (float)spatial;    /* divide by pixel count */

        /* Step 2: compute variance over all H×W pixels */
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

        /* Step 3: normalize each pixel and apply affine transform */
        float inv_std = 1.0f / sqrtf(var + EPS);   /* reciprocal of std-dev (with epsilon guard) */
        for (h = 0; h < h_dim; h++)
        {
            for (w = 0; w < w_dim; w++)
            {
                int idx = c * spatial + h * w_dim + w;   /* flat index */
                float xhat = (in[idx] - mean) * inv_std; /* normalized pixel */
                out[idx] = gamma[c] * xhat + beta[c];    /* affine: scale + shift */
            }
        }
    }
}

int main(void)
{
    /* ch0: all 1.0 (zero variance) — output must be all 0 */
    /* ch1: values 1..9 (mean=5)   — clear normalization    */
    float input[C * H * W] = {
        1.0f, 1.0f, 1.0f,   /* ch0 row0 */
        1.0f, 1.0f, 1.0f,   /* ch0 row1 */
        1.0f, 1.0f, 1.0f,   /* ch0 row2 */
        1.0f, 2.0f, 3.0f,   /* ch1 row0 */
        4.0f, 5.0f, 6.0f,   /* ch1 row1 */
        7.0f, 8.0f, 9.0f    /* ch1 row2 */
    };

    float output[C * H * W];

    /* identity affine: gamma=1, beta=0 for all channels */
    float gamma[C] = {1.0f, 1.0f};
    float beta[C]  = {0.0f, 0.0f};

    show("Input:", input, C, H, W);

    batch_norm(input, output, gamma, beta, C, H, W);

    show("Output (BatchNorm):", output, C, H, W);

    /* Correctness check:
     * ch0: all inputs identical → variance=0 → all outputs should be 0
     * ch1: mean of [1..9]=5, check output[ch1] has mean≈0 and var≈1 */
    float sum0 = 0.0f, sum1 = 0.0f;
    int i;
    for (i = 0; i < H * W; i++)
    {
        sum0 += output[i];                  /* ch0 outputs sum */
        sum1 += output[H * W + i];          /* ch1 outputs sum */
    }
    float mean0_out = sum0 / (float)(H * W);
    float mean1_out = sum1 / (float)(H * W);

    float var1 = 0.0f;
    for (i = 0; i < H * W; i++)
    {
        float diff = output[H * W + i] - mean1_out;
        var1 += diff * diff;
    }
    var1 /= (float)(H * W);

    printf("\nCheck ch0: all outputs = %.4f (expect 0.0000)\n", output[0]);
    printf("Check ch1: mean=%.4f (expect ~0), var=%.4f (expect ~1)\n",
           mean1_out, var1);

    int pass = (fabsf(output[0]) < 1e-3f) &&
               (fabsf(mean1_out) < 1e-3f) &&
               (fabsf(var1 - 1.0f) < 0.05f);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
