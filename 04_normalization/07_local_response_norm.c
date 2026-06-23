/*
 * Operator  : Local Response Normalization (cross-channel LRN)
 * Normalizes over: a sliding window of adjacent channels
 * Formula   : y[i] = x[i] / (k + alpha * sum_{j=max(0,i-n/2)}^{min(C-1,i+n/2)} x[j]^2)^beta
 * Models    : AlexNet, early VGG — largely obsolete but ONNX still has an LRN op
 * DEMO      : 5 channels [1,2,3,4,5], window size n=3
 *             middle channels (higher values, more neighbors) suppressed more than edges
 * Build     : gcc -O2 -o local_response_norm 07_local_response_norm.c -lm
 */

#include <stdio.h>
#include <math.h>

#define C     5     /* number of channels in demo */
#define H     1     /* spatial H=1 to isolate cross-channel effect */
#define W     1     /* spatial W=1 to isolate cross-channel effect */

/* LRN hyper-parameters (AlexNet used these values) */
#define LRN_N     3       /* window size (number of adjacent channels) */
#define LRN_K     2.0f    /* bias constant k */
#define LRN_ALPHA 1e-4f   /* scale factor alpha */
#define LRN_BETA  0.75f   /* exponent beta */

/* Print a [C][H][W] tensor as a flat channel vector (H=W=1 so one value per channel) */
static void show(const char *title, const float *m, int c_dim, int h_dim, int w_dim)
{
    (void)h_dim; (void)w_dim;
    int c;
    printf("%s\n  channels: [ ", title);
    for (c = 0; c < c_dim; c++)
    {
        /* flat index: c * H*W + 0 + 0 = c (since H=W=1) */
        printf("%10.6f", m[c]);
    }
    printf(" ]\n");
}

/*
 * local_response_norm: suppress each channel response by nearby channel energy.
 *
 * LRN was AlexNet's normalization before BN existed — it competes across
 * nearby channels; ONNX still has an LRN op.
 *
 * The window for channel i spans:
 *   j = max(0, i - floor(n/2))  ..  min(C-1, i + floor(n/2))
 *
 * in    : [C][H][W] input
 * out   : [C][H][W] output
 */
static void local_response_norm(const float *in, float *out,
                                int c_dim, int h_dim, int w_dim,
                                int n, float k, float alpha, float beta)
{
    int c, h, w, j;
    int spatial = h_dim * w_dim;   /* pixels per channel */
    int half_n  = n / 2;           /* half window size (integer division) */

    /* Step 1 + Step 2 (combined for LRN — no global mean/var; local sum of squares) */
    /* For each spatial position, compute the local energy window per channel */
    for (h = 0; h < h_dim; h++)
    {
        for (w = 0; w < w_dim; w++)
        {
            /* Step 1: for each channel i, compute sum of squared values in channel window */
            for (c = 0; c < c_dim; c++)
            {
                float sum_sq = 0.0f;

                int j_start = c - half_n;   /* window lower bound (before clamping) */
                int j_end   = c + half_n;   /* window upper bound (before clamping) */

                if (j_start < 0)         { j_start = 0; }         /* clamp to channel 0 */
                if (j_end   >= c_dim)    { j_end   = c_dim - 1; } /* clamp to last channel */

                for (j = j_start; j <= j_end; j++)
                {
                    float xj = in[j * spatial + h * w_dim + w];   /* neighbor channel value */
                    sum_sq += xj * xj;                             /* accumulate x[j]^2 */
                }

                /* Step 2: compute the normalization denominator */
                float denom = powf(k + alpha * sum_sq, beta);   /* (k + alpha * sum_sq)^beta */

                /* Step 3: divide input by denominator (no affine in LRN) */
                int idx = c * spatial + h * w_dim + w;   /* flat index of current element */
                out[idx] = in[idx] / denom;               /* suppressed output */
            }
        }
    }
}

int main(void)
{
    /* Single-pixel channels: values [1,2,3,4,5]
     * Channels with higher index have larger values AND more high-energy neighbors
     * → middle/high channels suppressed more than edge channel 0 */
    float input[C * H * W] = {
        1.0f,   /* ch0 */
        2.0f,   /* ch1 */
        3.0f,   /* ch2 */
        4.0f,   /* ch3 */
        5.0f    /* ch4 */
    };

    float output[C * H * W];

    show("Input:", input, C, H, W);
    printf("  LRN params: n=%d, k=%.1f, alpha=%.0e, beta=%.2f\n",
           LRN_N, LRN_K, (double)LRN_ALPHA, LRN_BETA);

    local_response_norm(input, output, C, H, W, LRN_N, LRN_K, LRN_ALPHA, LRN_BETA);

    show("Output (LRN):", output, C, H, W);

    /* Correctness: each output must be less than its input (normalization suppresses),
     * and the ratio out/in should decrease as channel index increases
     * (more neighbors → more suppression for middle/upper channels) */
    int c;
    printf("\nSuppression ratios (out/in, should generally decrease):\n  [ ");
    for (c = 0; c < C; c++)
    {
        printf("%8.6f", output[c] / input[c]);
    }
    printf(" ]\n");

    /* Check: all outputs < inputs (all ratios < 1) */
    int pass = 1;
    for (c = 0; c < C; c++)
    {
        if (output[c] >= input[c])   /* output must be suppressed */
        {
            pass = 0;
        }
    }

    /* Check: ch4 (most energy around it) suppressed more than ch0 (edge) */
    float ratio0 = output[0] / input[0];
    float ratio4 = output[4] / input[4];
    if (ratio4 >= ratio0)   /* inner channel should have a lower ratio than edge */
    {
        pass = 0;
    }

    printf("Check: ratio[0]=%.6f > ratio[4]=%.6f (edge suppressed less than inner)\n",
           ratio0, ratio4);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
