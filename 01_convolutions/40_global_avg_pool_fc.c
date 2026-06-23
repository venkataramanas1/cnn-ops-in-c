/* ============================================================================
 * [40] Global Avg Pool -> FC  --  classification head
 * ----------------------------------------------------------------------------
 * SHAPE: the entire spatial map gets COLLAPSED to a single number per channel
 * (one value per C, no H, no W). Then a fully-connected layer maps that vector
 * to class scores. From the top: the entire spatial grid shrinks to a single
 * point, then a linear transform maps it to class logits.
 *
 *   Input : [C][H][W]
 *   GAP   : average over H*W -> [C]     (no weight, no spatial footprint)
 *   Weight: [num_classes][C]             (the FC layer)
 *   Bias  : [num_classes]
 *   Output: [num_classes]                (class logits)
 *
 * FLOPs: C*H*W (the avg) + 2*C*num_classes (the FC)
 *
 * DEMO: a 4-channel feature map with spatially varying values. GAP reduces each
 * channel to its mean, then FC maps to 3 class scores. Apply softmax to get
 * probabilities. Print the whole pipeline: map -> vector -> logits -> probs.
 *
 * Build:  gcc 40_global_avg_pool_fc.c -o 40_global_avg_pool_fc -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s  [C=%d H=%d W=%d]\n", title, C, H, W);
    for (int c = 0; c < C; c++) {
        if (C > 1) { printf("  channel %d:\n", c); }
        for (int h = 0; h < H; h++) {
            printf("   ");
            for (int w = 0; w < W; w++) { printf("%6.2f", m[c*H*W+h*W+w]); }
            printf("\n");
        }
    }
    printf("\n");
}

static void show_vec(const char *title, const float *v, int n)
{
    printf("%s  [%d]\n   ", title, n);
    for (int i=0;i<n;i++) { printf("%8.4f", v[i]); }
    printf("\n\n");
}

int main(void)
{
    int C = 4, H = 3, W = 3, num_classes = 3;

    float *feat    = calloc(C*H*W,            sizeof(float));
    float *gap_vec = calloc(C,                sizeof(float));
    float *fc_w    = calloc(num_classes*C,    sizeof(float));
    float *fc_b    = calloc(num_classes,      sizeof(float));
    float *logits  = calloc(num_classes,      sizeof(float));
    float *probs   = calloc(num_classes,      sizeof(float));

    /* feature map: channel c has all values = (c+1) * ramp position */
    for (int c=0;c<C;c++) {
        for (int h=0;h<H;h++) {
            for (int x=0;x<W;x++) {
                feat[c*H*W+h*W+x] = (float)(c+1) + 0.1f*(float)(h*W+x);
            }
        }
    }

    /* FC weights: class k is sensitive to channels differently */
    /* class0 likes ch0 (w=1) ; class1 likes ch1,ch2 ; class2 likes ch3 */
    fc_w[0*C+0]=1.0f; fc_w[0*C+1]=0.1f; fc_w[0*C+2]=0.1f; fc_w[0*C+3]=0.0f;
    fc_w[1*C+0]=0.0f; fc_w[1*C+1]=1.0f; fc_w[1*C+2]=1.0f; fc_w[1*C+3]=0.0f;
    fc_w[2*C+0]=0.0f; fc_w[2*C+1]=0.0f; fc_w[2*C+2]=0.1f; fc_w[2*C+3]=2.0f;
    fc_b[0]=0.0f; fc_b[1]=0.0f; fc_b[2]=0.0f;

    /* --- global average pool ---
       Key mechanic: collapse the entire H*W spatial map for each channel into
       a single scalar (the spatial mean). This makes the classification head
       location-invariant and reduces C*H*W activations down to just C values. */
    for (int c=0;c<C;c++) {             /* c: channel */
        float s=0.0f;
        /* accumulate sum over all H*W spatial positions for channel c */
        for (int i=0;i<H*W;i++) { s+=feat[c*H*W+i]; } /* flat spatial index: c*H*W + i */
        gap_vec[c] = s / (float)(H*W);  /* gap_vec[c] = spatial mean of channel c */
    }

    /* --- FC layer ---
       Linear projection: logit[k] = bias[k] + sum_c( gap_vec[c] * fc_w[k][c] )
       Each class k has a weight vector of length C that scores how much each
       channel's average activation supports that class. */
    for (int k=0;k<num_classes;k++) {   /* k: class index */
        float s = fc_b[k];              /* start from bias for class k */
        /* weight flat index: [k][c] -> k*C + c */
        for (int c=0;c<C;c++) { s += gap_vec[c] * fc_w[k*C+c]; } /* dot product of gap vector with class row */
        logits[k] = s;                  /* raw class score (logit) */
    }

    /* --- softmax ---
       Converts raw logits to probabilities: prob[k] = exp(logit[k]) / sum_j exp(logit[j])
       Subtract maxv before exp for numerical stability (avoids overflow). */
    float maxv = logits[0];
    for (int k=1;k<num_classes;k++) { if (logits[k]>maxv) maxv=logits[k]; } /* find max logit for stability */
    float sum=0.0f;
    for (int k=0;k<num_classes;k++) { probs[k]=expf(logits[k]-maxv); sum+=probs[k]; } /* unnormalized exp */
    for (int k=0;k<num_classes;k++) { probs[k]/=sum; } /* normalize: probs now sum to 1.0 */

    show("INPUT  feature map (spatially varying, 4 channels)", feat, C, H, W);
    show_vec("AFTER GAP  (one scalar per channel)", gap_vec, C);
    show_vec("FC weights (each row is a class: [4])", fc_w, num_classes);
    show_vec("LOGITS  (raw class scores)", logits, num_classes);
    show_vec("PROBS   (softmax, sum to 1.0)", probs, num_classes);

    float prob_sum=0.0f;
    for (int k=0;k<num_classes;k++) { prob_sum+=probs[k]; }
    printf("[40] check: prob sum = %.6f (expected 1.000000)\n", prob_sum);
    printf("[40] predicted class = %d\n",
           (probs[0]>probs[1] && probs[0]>probs[2]) ? 0 :
           (probs[1]>probs[2]) ? 1 : 2);

    free(feat);free(gap_vec);free(fc_w);free(fc_b);free(logits);free(probs);
    return 0;
}
