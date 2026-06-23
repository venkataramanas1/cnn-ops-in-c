/* ============================================================================
 * [04] Anchor box decode  --  YOLO/SSD/DETR spatial head post-processing
 * ----------------------------------------------------------------------------
 * Anchor decode maps the network's DELTA predictions back to absolute pixel
 * coordinates. exp(tw) ensures predicted width is always positive. This is
 * the final spatial decode in YOLO/SSD/DETR heads.
 *
 *   Given: anchor [cx, cy, w, h]  (prior / anchor box)
 *          delta  [tx, ty, tw, th] (network output offsets/log-scale)
 *
 *   Decode formulas:
 *     pred_cx = anchor_cx + tx * anchor_w      (shift center x by fraction of width)
 *     pred_cy = anchor_cy + ty * anchor_h      (shift center y by fraction of height)
 *     pred_w  = anchor_w  * exp(tw)            (scale width: exp keeps it positive)
 *     pred_h  = anchor_h  * exp(th)            (scale height)
 *
 *   Convert [cx,cy,w,h] -> [x1,y1,x2,y2]:
 *     x1 = pred_cx - pred_w/2
 *     y1 = pred_cy - pred_h/2
 *     x2 = pred_cx + pred_w/2
 *     y2 = pred_cy + pred_h/2
 *
 *   4D tensor layout: [N][A][4]  where A = number of anchors (dynamic)
 *   Stride: element [n][a][k] = n*A*4 + a*4 + k
 *
 * DYNAMIC SHAPES: A (num_anchors) is a runtime variable — YOLO heads can have
 *   3 anchors per cell, and the number of cells is determined at runtime by
 *   the image size. Memory is heap-allocated to N*A*4.
 *
 * MODELS: YOLOv5/v8 (anchor-based), SSD, Faster R-CNN RPN all use this decode.
 *   DETR uses a variant (no anchor, but still cx/cy/w/h -> x1/y1/x2/y2).
 *
 * Demo: N=1, A=3 anchors, each with a small predicted delta.
 *   Show anchor priors, show deltas, show decoded boxes.
 *   Check: decoded box with zero deltas == anchor itself (identity case).
 *
 * Build: gcc 04_anchor_decode.c -o 04_anchor_decode -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void)
{
    /* runtime shape variables */
    int N = 1;  /* batch size */
    int A = 3;  /* number of anchors -- dynamic at runtime */

    /* heap-allocate: [N][A][4] for anchors, deltas, and decoded boxes */
    float *anchors  = (float *)malloc((size_t)N * A * 4 * sizeof(float)); /* [cx,cy,w,h] */
    float *deltas   = (float *)malloc((size_t)N * A * 4 * sizeof(float)); /* [tx,ty,tw,th] */
    float *decoded  = (float *)malloc((size_t)N * A * 4 * sizeof(float)); /* [x1,y1,x2,y2] */

    /* --- anchor priors: [cx, cy, w, h] in pixel coordinates ---
       These are the prior box shapes, defined at grid cell centers. */
    /* anchor 0: small square, center at (32,32) */
    anchors[0*A*4 + 0*4 + 0] = 32.0f; /* cx */
    anchors[0*A*4 + 0*4 + 1] = 32.0f; /* cy */
    anchors[0*A*4 + 0*4 + 2] = 20.0f; /* w  */
    anchors[0*A*4 + 0*4 + 3] = 20.0f; /* h  */

    /* anchor 1: wide rectangle, center at (64,64) */
    anchors[0*A*4 + 1*4 + 0] = 64.0f; /* cx */
    anchors[0*A*4 + 1*4 + 1] = 64.0f; /* cy */
    anchors[0*A*4 + 1*4 + 2] = 60.0f; /* w  */
    anchors[0*A*4 + 1*4 + 3] = 30.0f; /* h  */

    /* anchor 2 (identity test): zero deltas should reproduce this exactly */
    anchors[0*A*4 + 2*4 + 0] = 100.0f; /* cx */
    anchors[0*A*4 + 2*4 + 1] = 100.0f; /* cy */
    anchors[0*A*4 + 2*4 + 2] =  40.0f; /* w  */
    anchors[0*A*4 + 2*4 + 3] =  40.0f; /* h  */

    /* --- network delta predictions: [tx, ty, tw, th] ---
       Small offsets: the network adjusts anchors slightly. */
    /* anchor 0: shift center slightly right and down, scale up 20% */
    deltas[0*A*4 + 0*4 + 0] =  0.1f;  /* tx: shift cx by 0.1 * anchor_w */
    deltas[0*A*4 + 0*4 + 1] =  0.1f;  /* ty: shift cy by 0.1 * anchor_h */
    deltas[0*A*4 + 0*4 + 2] =  0.2f;  /* tw: scale w by exp(0.2) ≈ 1.22 */
    deltas[0*A*4 + 0*4 + 3] = -0.1f;  /* th: scale h by exp(-0.1) ≈ 0.90 */

    /* anchor 1: shift center left, shrink width */
    deltas[0*A*4 + 1*4 + 0] = -0.2f;  /* tx */
    deltas[0*A*4 + 1*4 + 1] =  0.05f; /* ty */
    deltas[0*A*4 + 1*4 + 2] = -0.3f;  /* tw: exp(-0.3) ≈ 0.74 */
    deltas[0*A*4 + 1*4 + 3] =  0.1f;  /* th */

    /* anchor 2: ZERO deltas -> decoded box should equal anchor (identity) */
    deltas[0*A*4 + 2*4 + 0] =  0.0f;
    deltas[0*A*4 + 2*4 + 1] =  0.0f;
    deltas[0*A*4 + 2*4 + 2] =  0.0f;
    deltas[0*A*4 + 2*4 + 3] =  0.0f;

    /* --- print anchor priors --- */
    printf("[04] Anchor decode: N=%d, A=%d anchors\n\n", N, A);
    printf("  Anchor priors [cx, cy, w, h]:\n");
    for (int a = 0; a < A; a++) {
        int base = 0*A*4 + a*4;  /* stride: n*A*4 + a*4 */
        printf("    anchor %d: cx=%6.1f  cy=%6.1f  w=%6.1f  h=%6.1f\n",
               a, anchors[base+0], anchors[base+1], anchors[base+2], anchors[base+3]);
    }

    printf("\n  Deltas [tx, ty, tw, th]:\n");
    for (int a = 0; a < A; a++) {
        int base = 0*A*4 + a*4;
        printf("    anchor %d: tx=%6.3f  ty=%6.3f  tw=%6.3f  th=%6.3f\n",
               a, deltas[base+0], deltas[base+1], deltas[base+2], deltas[base+3]);
    }
    printf("\n");

    /* --- decode: apply anchor decode formulas ---
       For each batch n and each anchor a, compute predicted [cx,cy,w,h] then
       convert to corner format [x1,y1,x2,y2]. */
    for (int n = 0; n < N; n++) {              /* n: batch index */
        for (int a = 0; a < A; a++) {          /* a: anchor index */
            int base = n*A*4 + a*4;            /* flat offset for this (n,a) */

            float acx = anchors[base+0];       /* anchor center x */
            float acy = anchors[base+1];       /* anchor center y */
            float aw  = anchors[base+2];       /* anchor width   */
            float ah  = anchors[base+3];       /* anchor height  */

            float tx  = deltas[base+0];        /* predicted delta for cx */
            float ty  = deltas[base+1];        /* predicted delta for cy */
            float tw  = deltas[base+2];        /* predicted log-scale for w */
            float th  = deltas[base+3];        /* predicted log-scale for h */

            /* decode center-form prediction */
            float pcx = acx + tx * aw;         /* pred_cx = anchor_cx + tx * anchor_w */
            float pcy = acy + ty * ah;         /* pred_cy = anchor_cy + ty * anchor_h */
            float pw  = aw  * expf(tw);        /* pred_w = anchor_w * exp(tw)  [always positive] */
            float ph  = ah  * expf(th);        /* pred_h = anchor_h * exp(th)  [always positive] */

            /* convert center-form [cx,cy,w,h] -> corner-form [x1,y1,x2,y2] */
            decoded[base+0] = pcx - pw * 0.5f; /* x1 = pred_cx - pred_w/2 */
            decoded[base+1] = pcy - ph * 0.5f; /* y1 = pred_cy - pred_h/2 */
            decoded[base+2] = pcx + pw * 0.5f; /* x2 = pred_cx + pred_w/2 */
            decoded[base+3] = pcy + ph * 0.5f; /* y2 = pred_cy + pred_h/2 */
        }
    }

    /* --- print decoded boxes --- */
    printf("  Decoded boxes [x1, y1, x2, y2]:\n");
    for (int a = 0; a < A; a++) {
        int base = 0*A*4 + a*4;
        printf("    anchor %d: x1=%7.3f  y1=%7.3f  x2=%7.3f  y2=%7.3f\n",
               a, decoded[base+0], decoded[base+1], decoded[base+2], decoded[base+3]);
    }
    printf("\n");

    /* --- identity check: anchor 2 had zero deltas ---
       exp(0)=1 so pred_w=anchor_w, pred_h=anchor_h; tx=ty=0 so cx/cy unchanged.
       Expected x1 = 100-20=80, y1=80, x2=120, y2=120. */
    int base2 = 0*A*4 + 2*4;
    float x1_id = decoded[base2+0];
    float y1_id = decoded[base2+1];
    float x2_id = decoded[base2+2];
    float y2_id = decoded[base2+3];
    printf("[04] check (identity): anchor2 zero-delta decoded = [%.1f,%.1f,%.1f,%.1f]\n",
           x1_id, y1_id, x2_id, y2_id);
    printf("[04] check (identity): expected [80.0,80.0,120.0,120.0] -> %s\n",
           (fabsf(x1_id-80.0f)<0.001f && fabsf(y1_id-80.0f)<0.001f &&
            fabsf(x2_id-120.0f)<0.001f && fabsf(y2_id-120.0f)<0.001f) ? "PASS" : "FAIL");

    free(anchors); free(deltas); free(decoded);
    return 0;
}
