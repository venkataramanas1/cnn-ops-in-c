/* ============================================================================
 * [38] FPN Lateral 1x1  --  Feature Pyramid Network channel alignment
 * ----------------------------------------------------------------------------
 * SHAPE: a single dot. Before two feature maps at different scales can be
 * ADDED together in the FPN top-down pathway, their channel counts must match.
 * The lateral 1x1 aligns them. No spatial awareness needed -- just channel
 * projection at each spatial location.
 *
 *   Input : [C_in][H][W]             (deep feature map, many channels)
 *   Weight: [C_fpn][C_in][1][1]      (C_fpn is the uniform FPN channel count)
 *   Output: [C_fpn][H][W]
 *
 * FLOPs: 2 * C_in * C_fpn * H * W
 *
 * DEMO: two feature maps at different scales (stride 8 and stride 16) with
 * different channel counts are aligned to the same C_fpn=4 by the 1x1 lateral.
 * The aligned maps are then upsampled and added -- the core FPN merge.
 *
 * Build:  gcc 38_fpn_lateral_1x1.c -o 38_fpn_lateral_1x1 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

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

static void conv1x1(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    /* FPN lateral 1x1: aligns channel counts across scales so features can be
       element-wise added in the top-down pathway. No spatial context -- each
       pixel independently remapped from C_in to C_fpn channels. */
    for (int co=0;co<Co;co++) {         /* co: output (FPN) channel */
        for (int h=0;h<H;h++) {         /* h: spatial row */
            for (int x=0;x<W;x++) {     /* x: spatial column */
                float s=0.0f;           /* accumulator for this output pixel */
                for (int ci=0;ci<Ci;ci++) { /* ci: input channel */
                    /* input flat index: [ci][h][x] -> ci*H*W + h*W + x */
                    /* weight flat index: [co][ci] -> co*Ci + ci  (1x1: no kh/kw) */
                    s += in[ci*H*W+h*W+x] * w[co*Ci+ci];
                }
                out[co*H*W+h*W+x]=s; /* aligned feature at [co][h][x] */
            }
        }
    }
}

/* nearest-neighbor 2x upsample: each input pixel maps to a 2x2 block of
   identical values in the output. Used in FPN top-down pathway to bring
   a coarser (deeper) feature map up to the spatial size of the next level. */
static void upsample2x(const float *in, float *out, int C, int H, int W)
{
    int H2=H*2, W2=W*2;                 /* output spatial dims after 2x upsample */
    for (int c=0;c<C;c++) {             /* c: channel (unchanged by upsample) */
        for (int h=0;h<H;h++) {         /* h: input row */
            for (int w=0;w<W;w++) {     /* w: input column */
                float v = in[c*H*W+h*W+w]; /* input pixel value replicated to 2x2 block */
                /* top-left of 2x2 block */
                out[c*H2*W2+(h*2)*W2+(w*2)]   = v;
                /* top-right of 2x2 block */
                out[c*H2*W2+(h*2)*W2+(w*2+1)] = v;
                /* bottom-left of 2x2 block */
                out[c*H2*W2+(h*2+1)*W2+(w*2)] = v;
                /* bottom-right of 2x2 block */
                out[c*H2*W2+(h*2+1)*W2+(w*2+1)]=v;
            }
        }
    }
}

int main(void)
{
    /* stride-16 map: small spatial, many channels */
    int Cs16=8, Hs16=2, Ws16=2;
    /* stride-8  map: larger spatial, fewer channels */
    int Cs8 =6, Hs8 =4, Ws8 =4;
    int C_fpn = 4;

    float *feat_s16   = calloc(Cs16*Hs16*Ws16, sizeof(float));
    float *lat_w_s16  = calloc(C_fpn*Cs16,     sizeof(float));
    float *aligned_s16= calloc(C_fpn*Hs16*Ws16,sizeof(float));
    float *up_s16     = calloc(C_fpn*Hs8*Ws8,  sizeof(float));

    float *feat_s8    = calloc(Cs8*Hs8*Ws8,    sizeof(float));
    float *lat_w_s8   = calloc(C_fpn*Cs8,      sizeof(float));
    float *aligned_s8 = calloc(C_fpn*Hs8*Ws8,  sizeof(float));

    float *merged     = calloc(C_fpn*Hs8*Ws8,  sizeof(float));

    for (int i=0;i<Cs16*Hs16*Ws16;i++) { feat_s16[i]=2.0f; }
    for (int i=0;i<Cs8 *Hs8 *Ws8 ;i++) { feat_s8[i] =1.0f; }
    for (int i=0;i<C_fpn*Cs16;i++) { lat_w_s16[i]=1.0f/Cs16; }
    for (int i=0;i<C_fpn*Cs8 ;i++) { lat_w_s8 [i]=1.0f/Cs8;  }

    conv1x1(feat_s16, lat_w_s16, aligned_s16, Cs16, C_fpn, Hs16, Ws16);
    upsample2x(aligned_s16, up_s16, C_fpn, Hs16, Ws16);
    conv1x1(feat_s8,  lat_w_s8,  aligned_s8,  Cs8,  C_fpn, Hs8,  Ws8);

    for (int i=0;i<C_fpn*Hs8*Ws8;i++) { merged[i]=up_s16[i]+aligned_s8[i]; }

    show("FEAT stride-16 (2x2, 8ch)", feat_s16, Cs16, Hs16, Ws16);
    show("LATERAL-ALIGNED stride-16 (2x2, 4ch)", aligned_s16, C_fpn, Hs16, Ws16);
    show("UPSAMPLED stride-16 (4x4, 4ch)", up_s16, C_fpn, Hs8, Ws8);
    show("FEAT stride-8 (4x4, 6ch)", feat_s8, Cs8, Hs8, Ws8);
    show("LATERAL-ALIGNED stride-8 (4x4, 4ch)", aligned_s8, C_fpn, Hs8, Ws8);
    show("MERGED = up_s16 + aligned_s8 (the FPN add)", merged, C_fpn, Hs8, Ws8);

    printf("[38] check: merged[0][0] = %.2f (up_s16=%.2f + aligned_s8=%.2f)\n",
           merged[0], up_s16[0], aligned_s8[0]);

    free(feat_s16);free(lat_w_s16);free(aligned_s16);free(up_s16);
    free(feat_s8);free(lat_w_s8);free(aligned_s8);free(merged);
    return 0;
}
