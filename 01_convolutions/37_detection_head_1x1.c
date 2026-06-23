/* ============================================================================
 * [37] Detection Head 1x1
 * ----------------------------------------------------------------------------
 * SHAPE: a single dot. Projects the feature map to exactly
 * (num_anchors * (5 + num_classes)) channels. No spatial awareness -- pure
 * channel projection at each spatial location. In YOLOv3 the grid cell at (h,w)
 * directly predicts bounding boxes; this 1x1 is the final output layer.
 *
 *   Input : [C_in][H][W]              (deep feature map from backbone)
 *   Weight: [A*(5+cls)][C_in][1][1]   (A anchors, 5=tx,ty,tw,th,conf)
 *   Output: [A*(5+cls)][H][W]         (raw predictions, one per grid cell)
 *
 * FLOPs: 2 * C_in * A*(5+cls) * H * W
 *
 * DEMO: a 4x4 feature map, 3 anchors, 2 classes -> 3*(5+2)=21 output channels.
 * Each spatial position outputs 21 raw prediction values. Print the raw output
 * for one grid cell to see all 21 values (tx,ty,tw,th,conf,cls0,cls1 per anchor).
 *
 * Build:  gcc 37_detection_head_1x1.c -o 37_detection_head_1x1 -lm
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
            for (int w = 0; w < W; w++) { printf("%7.3f", m[c*H*W+h*W+w]); }
            printf("\n");
        }
    }
    printf("\n");
}

static void conv1x1(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    /* detection head: 1x1 conv projects C_in backbone features to
       A*(5+cls) output channels at each spatial location (h, x).
       No spatial neighbourhood -- each grid cell is projected independently.
       Output layout per cell: [tx, ty, tw, th, conf, cls0..cls_{n-1}]
       repeated for each anchor A.
       Regression branch:     output channels base+0..base+3 = tx,ty,tw,th offsets.
       Classification branch: output channel  base+4 = conf, base+5..base+4+cls = class scores. */
    for (int co=0;co<Co;co++) {         /* co: output channel (encodes anchor + field index) */
        for (int h=0;h<H;h++) {         /* h: grid row */
            for (int x=0;x<W;x++) {     /* x: grid column */
                float s=0.0f;           /* accumulator for this output channel at grid cell (h,x) */
                for (int ci=0;ci<Ci;ci++) { /* ci: input channel (backbone feature) */
                    /* input flat index: [ci][h][x] -> ci*H*W + h*W + x */
                    /* weight flat index: [co][ci] -> co*Ci + ci  (no kh/kw: 1x1 kernel) */
                    s += in[ci*H*W+h*W+x] * w[co*Ci+ci];
                }
                out[co*H*W+h*W+x]=s; /* store raw prediction for channel co at grid cell (h,x) */
            }
        }
    }
}

int main(void)
{
    int C_in  = 8;
    int A     = 3;           /* anchors per grid cell */
    int cls   = 2;           /* number of classes */
    int C_out = A * (5 + cls);   /* = 21 */
    int H = 4, W = 4;

    float *in  = calloc(C_in *H*W,  sizeof(float));
    float *w   = calloc(C_out*C_in, sizeof(float));
    float *out = calloc(C_out*H*W,  sizeof(float));

    /* feature map: each channel a ramp */
    for (int c=0;c<C_in;c++) {
        for (int i=0;i<H*W;i++) { in[c*H*W+i]=(float)(c+1)*0.1f; }
    }
    /* random-ish weights */
    for (int i=0;i<C_out*C_in;i++) {
        w[i] = ((i % 5) - 2) * 0.1f;
    }

    show("INPUT  (feature map, 8 channels)", in, C_in, H, W);
    conv1x1(in, w, out, C_in, C_out, H, W);

    printf("OUTPUT has %d channels = %d anchors x (5 + %d classes)\n",
           C_out, A, cls);
    printf("Showing predictions for grid cell (0,0) -- all %d values:\n\n", C_out);

    for (int a=0;a<A;a++) {
        int base = a*(5+cls);
        printf("  Anchor %d: tx=%.3f  ty=%.3f  tw=%.3f  th=%.3f  conf=%.3f  "
               "cls0=%.3f  cls1=%.3f\n",
               a,
               out[(base+0)*H*W], out[(base+1)*H*W],
               out[(base+2)*H*W], out[(base+3)*H*W],
               out[(base+4)*H*W],
               out[(base+5)*H*W], out[(base+6)*H*W]);
    }

    printf("\n[37] check: output shape = [%d][%d][%d] (%d * (5+%d))\n",
           C_out, H, W, A, cls);

    free(in); free(w); free(out);
    return 0;
}
