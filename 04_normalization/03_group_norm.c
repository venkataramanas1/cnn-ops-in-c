/*
 * Operator  : Group Normalization
 * Normalizes over: groups of C/G channels, each group normalized over
 *   (C/G) * H * W values
 * Models    : Mask R-CNN (GN for small batch object detection),
 *             π0 diffusion VLA (GN in UNet backbone)
 * DEMO      : C=4, H=3, W=3, G=2 (2 groups of 2 channels each)
 *             input with very different scales per channel to show grouping
 * Build     : gcc -O2 -o group_norm 03_group_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define C   4
#define H   3
#define W   3
#define G   2     /* number of groups */
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
 * group_norm: normalize over G groups of C/G channels.
 *
 * GroupNorm is batch-size independent — critical for detection (small batches)
 * and diffusion VLA policy networks.
 *
 * in    : [C][H][W] input
 * out   : [C][H][W] output
 * gamma : [C] per-channel scale
 * beta  : [C] per-channel shift
 */
static void group_norm(const float *in, float *out,
                       const float *gamma, const float *beta,
                       int c_dim, int h_dim, int w_dim, int g)
{
    int cg, c, h, w;
    int channels_per_group = c_dim / g;      /* e.g. 4/2 = 2 channels per group */
    int spatial = h_dim * w_dim;             /* pixels per channel */
    int group_size = channels_per_group * spatial; /* elements per group */

    for (cg = 0; cg < g; cg++)              /* iterate over each group */
    {
        int c_start = cg * channels_per_group;  /* first channel index in this group */

        /* Step 1: compute mean over all channels and spatial positions in the group */
        float mean = 0.0f;
        for (c = c_start; c < c_start + channels_per_group; c++)
        {
            for (h = 0; h < h_dim; h++)
            {
                for (w = 0; w < w_dim; w++)
                {
                    mean += in[c * spatial + h * w_dim + w];   /* accumulate element */
                }
            }
        }
        mean /= (float)group_size;   /* divide by total elements in group */

        /* Step 2: compute variance over the same set of elements */
        float var = 0.0f;
        for (c = c_start; c < c_start + channels_per_group; c++)
        {
            for (h = 0; h < h_dim; h++)
            {
                for (w = 0; w < w_dim; w++)
                {
                    float diff = in[c * spatial + h * w_dim + w] - mean;   /* deviation */
                    var += diff * diff;   /* squared deviation */
                }
            }
        }
        var /= (float)group_size;   /* average squared deviation */

        /* Step 3: normalize each element in the group and apply per-channel affine */
        float inv_std = 1.0f / sqrtf(var + EPS);   /* 1/std with epsilon guard */
        for (c = c_start; c < c_start + channels_per_group; c++)
        {
            for (h = 0; h < h_dim; h++)
            {
                for (w = 0; w < w_dim; w++)
                {
                    int idx = c * spatial + h * w_dim + w;        /* flat index */
                    float xhat = (in[idx] - mean) * inv_std;      /* normalized */
                    out[idx] = gamma[c] * xhat + beta[c];         /* affine per-channel */
                }
            }
        }
    }
}

int main(void)
{
    /* ch0,ch1 small values (~1); ch2,ch3 large values (~100) to show grouping effect */
    float input[C * H * W] = {
        /* ch0 (group0) */
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        /* ch1 (group0) */
        2.0f, 4.0f, 6.0f,
        8.0f, 10.0f, 12.0f,
        14.0f, 16.0f, 18.0f,
        /* ch2 (group1) */
        100.0f, 200.0f, 300.0f,
        400.0f, 500.0f, 600.0f,
        700.0f, 800.0f, 900.0f,
        /* ch3 (group1) */
        50.0f, 100.0f, 150.0f,
        200.0f, 250.0f, 300.0f,
        350.0f, 400.0f, 450.0f
    };

    float output[C * H * W];

    /* identity affine: gamma=1, beta=0 for all channels */
    float gamma[C] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[C]  = {0.0f, 0.0f, 0.0f, 0.0f};

    show("Input:", input, C, H, W);

    group_norm(input, output, gamma, beta, C, H, W, G);

    show("Output (GroupNorm G=2):", output, C, H, W);

    /* Correctness: mean of each group's output should be ~0 */
    int g, c, i;
    int channels_per_group = C / G;
    int spatial = H * W;
    int pass = 1;

    for (g = 0; g < G; g++)
    {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int c_start = g * channels_per_group;
        int count = channels_per_group * spatial;

        for (c = c_start; c < c_start + channels_per_group; c++)
        {
            for (i = 0; i < spatial; i++)
            {
                float v = output[c * spatial + i];
                sum    += v;
                sum_sq += v * v;
            }
        }
        float mean_out = sum / (float)count;
        float var_out  = sum_sq / (float)count - mean_out * mean_out;

        printf("\nGroup %d: mean=%.5f (expect ~0), var=%.5f (expect ~1)\n",
               g, mean_out, var_out);

        if (fabsf(mean_out) > 1e-3f || fabsf(var_out - 1.0f) > 0.05f)
        {
            pass = 0;
        }
    }

    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
