/* ============================================================================
 * [31] CBAM Spatial Attention
 * ----------------------------------------------------------------------------
 * SHAPE: compress C channels down to 2 (channel-avg + channel-max), then apply
 * a single Conv 7x7. The 7x7 square operates on a 2-channel map to produce a
 * 1-channel spatial attention mask, which is Sigmoid-gated and multiplied back.
 * It tells the network WHERE to look (spatial) -- [30] told it WHAT to weight
 * (channel). In CBAM they are applied in sequence: channel first, then spatial.
 *
 *   Input : [C][H][W]
 *   Compress: avg over C -> [1][H][W],  max over C -> [1][H][W]
 *   Concat  : [2][H][W]
 *   Weight  : [1][2][7][7]              (one 7x7 kernel on 2-channel map)
 *   Output  : [C][H][W]  = input * sigmoid(conv7x7([avg_map ; max_map]))
 *
 * FLOPs: H*W*2*49  (conv7x7 on a tiny 2-channel map)
 *
 * DEMO: a 4-channel input with a single bright pixel in the top-left. The
 * spatial attention mask lights up around that bright region.
 *
 * Build:  gcc 31_cbam_spatial.c -o 31_cbam_spatial -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K   7
#define PAD 3

static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s  [C=%d H=%d W=%d]\n", title, C, H, W);
    for (int c = 0; c < C; c++) {
        if (C > 1) {
            printf("  channel %d:\n", c);
        }
        for (int h = 0; h < H; h++) {
            printf("   ");
            for (int w = 0; w < W; w++) {
                printf("%7.4f", m[c*H*W + h*W + w]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

int main(void)
{
    int C = 4, H = 5, W = 5;

    float *in     = calloc(C*H*W,   sizeof(float));
    float *avg_map= calloc(H*W,     sizeof(float));
    float *max_map= calloc(H*W,     sizeof(float));
    float *cat2   = calloc(2*H*W,   sizeof(float));
    float *wconv  = calloc(2*K*K,   sizeof(float));
    float *raw_attn=calloc(H*W,     sizeof(float));
    float *attn   = calloc(H*W,     sizeof(float));
    float *out    = calloc(C*H*W,   sizeof(float));

    /* input: bright spot at (0,0) in all channels */
    for (int c = 0; c < C; c++) {
        in[c*H*W + 0] = 4.0f;
        for (int i = 1; i < H*W; i++) {
            in[c*H*W + i] = 0.5f;
        }
    }

    /* step 1: compress channel dim -- for each spatial position compute avg and max
     * across all C channels, collapsing [C][H][W] to two [1][H][W] descriptor maps */
    for (int i = 0; i < H*W; i++) {              /* i: flat spatial index (0..H*W-1) */
        float s = 0.0f, m = in[i];               /* s: channel sum; m: channel max */
        for (int c = 0; c < C; c++) {            /* c: channel axis being compressed */
            s += in[c*H*W + i];
            if (in[c*H*W + i] > m) { m = in[c*H*W + i]; } /* track max across channels at this position */
        }
        avg_map[i] = s / (float)C;              /* avg_map[i]: mean over channels at position i */
        max_map[i] = m;                          /* max_map[i]: peak channel activation at position i */
    }

    /* step 2: concatenate avg and max maps to a 2-channel spatial descriptor [2][H][W] */
    for (int i = 0; i < H*W; i++) {
        cat2[0*H*W + i] = avg_map[i];            /* channel 0 = avg descriptor */
        cat2[1*H*W + i] = max_map[i];            /* channel 1 = max descriptor */
    }

    /* 7x7 conv kernel weights: all equal (spatial averaging over wide receptive field) */
    for (int i = 0; i < 2*K*K; i++) {
        wconv[i] = 1.0f / (float)(2 * K * K);   /* average over neighborhood */
    }

    /* step 3: conv7x7 (2->1 channel) -- large receptive field learns WHERE attention should focus */
    for (int h = 0; h < H; h++) {                /* h: output row */
        for (int x = 0; x < W; x++) {           /* x: output col */
            float sum = 0.0f;
            for (int cc = 0; cc < 2; cc++) {     /* cc: input channel (0=avg map, 1=max map) */
                for (int kh = 0; kh < K; kh++) {     /* kh: kernel row (0..K-1) */
                    for (int kw = 0; kw < K; kw++) { /* kw: kernel col (0..K-1) */
                        int ih = h + kh - PAD;        /* map to input coords with padding offset */
                        int iw = x + kw - PAD;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            continue;                 /* outside image boundary = zero padding */
                        }
                        /* input flat index: [cc][ih][iw] -> cc*H*W + ih*W + iw */
                        /* weight flat index: [cc][kh][kw] -> cc*K*K + kh*K + kw */
                        sum += cat2[cc*H*W + ih*W + iw] * wconv[cc*K*K + kh*K + kw];
                    }
                }
            }
            raw_attn[h*W + x] = sum;
            attn    [h*W + x] = 1.0f / (1.0f + expf(-sum));  /* sigmoid: convert score to [0,1] attention weight */
        }
    }

    /* step 4: scale all channels by the spatial attention mask (broadcast over C) */
    for (int c = 0; c < C; c++) {               /* c: channel to modulate */
        for (int i = 0; i < H*W; i++) {         /* i: flat spatial index */
            out[c*H*W + i] = in[c*H*W + i] * attn[i]; /* attn[i] is the same mask for every channel */
        }
    }

    show("INPUT  (bright spot at [0][0])", in, C, H, W);
    show("AVG MAP (channel average)", avg_map, 1, H, W);
    show("MAX MAP (channel max)", max_map, 1, H, W);
    show("SPATIAL ATTENTION (sigmoid of 7x7 conv -- higher near bright spot)", attn, 1, H, W);
    show("OUTPUT (input * spatial attention)", out, C, H, W);

    printf("[31] check: attn[0][0] > attn[2][2] (bright corner > center)\n");
    printf("[31] attn[0][0]=%.4f  attn[2][2]=%.4f\n", attn[0], attn[2*W+2]);

    free(in);free(avg_map);free(max_map);free(cat2);
    free(wconv);free(raw_attn);free(attn);free(out);
    return 0;
}
