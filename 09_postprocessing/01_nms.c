/* ============================================================================
 * [01] Non-Maximum Suppression (NMS)  --  object detection post-processing
 * ----------------------------------------------------------------------------
 * NMS: greedily keep the highest-scoring box, then suppress nearby boxes
 * (IoU > thresh). The IoU threshold trades precision (low) vs recall (high).
 *
 *   Input : [N_boxes][5]  each row = [x1, y1, x2, y2, score]
 *   Output: kept[] boolean mask of size N_boxes; surviving box indices
 *
 *   IoU = intersection_area / union_area
 *   intersection: max(0, min(x2a,x2b)-max(x1a,x1b))
 *               * max(0, min(y2a,y2b)-max(y1a,y1b))
 *   union = area_a + area_b - intersection
 *
 * DYNAMIC SHAPES: N_boxes is a runtime variable (number of proposed detections).
 *   kept[] is a heap-allocated boolean mask; order[] is a runtime sort index.
 *
 * MODELS: YOLOv5/v8, SSD, Faster R-CNN all use NMS as their final decode step.
 *   In VLA detect-then-act pipelines (RT-2, SpatialVLA) NMS filters perceived
 *   objects before the action policy.
 *
 * Demo: 5 boxes in 2 obvious clusters. Boxes 0,1,2 overlap strongly; 3,4 overlap.
 *   scores=[0.9,0.85,0.8,0.75,0.7], threshold=0.5
 *   Expected: exactly 2 boxes kept (one per cluster).
 *
 * Build: gcc 01_nms.c -o 01_nms -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- helper: compute IoU between box i and box j in [N][5] array --- */
static float iou(const float *boxes, int i, int j)
{
    /* unpack box i: stride 5 per row */
    float x1a = boxes[i*5+0];
    float y1a = boxes[i*5+1];
    float x2a = boxes[i*5+2];
    float y2a = boxes[i*5+3];

    /* unpack box j */
    float x1b = boxes[j*5+0];
    float y1b = boxes[j*5+1];
    float x2b = boxes[j*5+2];
    float y2b = boxes[j*5+3];

    /* intersection rectangle: clamp negative overlap to 0 */
    float ix = fmaxf(0.0f, fminf(x2a, x2b) - fmaxf(x1a, x1b)); /* intersection width  */
    float iy = fmaxf(0.0f, fminf(y2a, y2b) - fmaxf(y1a, y1b)); /* intersection height */
    float inter = ix * iy;                                        /* intersection area   */

    /* individual box areas */
    float area_a = (x2a - x1a) * (y2a - y1a);  /* area of box i */
    float area_b = (x2b - x1b) * (y2b - y1b);  /* area of box j */

    float uni = area_a + area_b - inter;         /* union = a + b - overlap  */
    if (uni <= 0.0f) { return 0.0f; }            /* guard against degenerate boxes */
    return inter / uni;                           /* IoU in [0,1] */
}

/* --- comparator for qsort: sort order[] by descending score --- */
static const float *g_scores = NULL; /* pointer used by comparator */
static int cmp_desc(const void *a, const void *b)
{
    float sa = g_scores[*(const int *)a];  /* score at index a */
    float sb = g_scores[*(const int *)b];  /* score at index b */
    if (sb > sa) { return  1; }            /* descending: higher score first */
    if (sb < sa) { return -1; }
    return 0;
}

int main(void)
{
    /* --- input boxes [N_boxes][5] = [x1, y1, x2, y2, score] --- */
    int N_boxes = 5;  /* runtime variable: number of candidate detections */

    /* heap-allocate: N_boxes is dynamic */
    float *boxes = (float *)malloc((size_t)N_boxes * 5 * sizeof(float));
    int   *order = (int   *)malloc((size_t)N_boxes * sizeof(int));
    int   *kept  = (int   *)calloc((size_t)N_boxes,  sizeof(int)); /* boolean mask */

    /* cluster A: boxes 0,1,2 heavily overlapping (top-left region) */
    /* [x1, y1, x2, y2, score] */
    boxes[0*5+0]=10; boxes[0*5+1]=10; boxes[0*5+2]=50; boxes[0*5+3]=50; boxes[0*5+4]=0.90f;
    boxes[1*5+0]=12; boxes[1*5+1]=11; boxes[1*5+2]=52; boxes[1*5+3]=51; boxes[1*5+4]=0.85f;
    boxes[2*5+0]=11; boxes[2*5+1]=13; boxes[2*5+2]=49; boxes[2*5+3]=53; boxes[2*5+4]=0.80f;

    /* cluster B: boxes 3,4 overlapping (bottom-right region, no overlap with A) */
    boxes[3*5+0]=80; boxes[3*5+1]=80; boxes[3*5+2]=120; boxes[3*5+3]=120; boxes[3*5+4]=0.75f;
    boxes[4*5+0]=82; boxes[4*5+1]=81; boxes[4*5+2]=122; boxes[4*5+3]=121; boxes[4*5+4]=0.70f;

    float iou_thresh = 0.5f;  /* suppress boxes with IoU > 0.5 */

    /* --- print input boxes --- */
    printf("[01] NMS input boxes [N=%d][5] = [x1, y1, x2, y2, score]\n", N_boxes);
    for (int i = 0; i < N_boxes; i++) {
        printf("  box %d: x1=%5.1f y1=%5.1f x2=%5.1f y2=%5.1f score=%.2f\n",
               i,
               boxes[i*5+0], boxes[i*5+1],
               boxes[i*5+2], boxes[i*5+3],
               boxes[i*5+4]);
    }
    printf("\n");

    /* --- step 1: initialize order[] = [0,1,...,N-1] then sort by score desc --- */
    for (int i = 0; i < N_boxes; i++) { order[i] = i; }

    /* point global comparator pointer to the score column (column 4) */
    float *scores_col = (float *)malloc((size_t)N_boxes * sizeof(float));
    for (int i = 0; i < N_boxes; i++) { scores_col[i] = boxes[i*5+4]; } /* extract scores */
    g_scores = scores_col;
    qsort(order, (size_t)N_boxes, sizeof(int), cmp_desc);  /* sort indices by score desc */

    printf("[01] sorted order by score desc: ");
    for (int i = 0; i < N_boxes; i++) { printf("%d ", order[i]); }
    printf("\n\n");

    /* --- step 2: greedy NMS loop --- */
    /* kept[] starts all zero; mark each surviving box, then suppress overlapping ones */
    for (int i = 0; i < N_boxes; i++) {
        int idx = order[i];          /* current candidate (highest remaining score) */

        /* check if this candidate was already suppressed by a higher-scoring box */
        if (kept[idx] == -1) { continue; }  /* -1 means suppressed, skip */

        kept[idx] = 1;  /* mark as KEPT: this is a surviving detection */

        /* suppress all subsequent boxes (lower score) that overlap too much */
        for (int j = i + 1; j < N_boxes; j++) {
            int jdx = order[j];
            if (kept[jdx] == -1) { continue; }  /* already suppressed, skip */

            float iou_val = iou(boxes, idx, jdx);  /* compute IoU(kept box, candidate) */
            printf("  IoU(box%d, box%d) = %.4f", idx, jdx, iou_val);

            if (iou_val > iou_thresh) {
                kept[jdx] = -1;  /* suppress: too much overlap with a kept box */
                printf("  -> SUPPRESS box%d", jdx);
            }
            printf("\n");
        }
    }
    printf("\n");

    /* --- print results --- */
    printf("[01] NMS result (threshold=%.2f):\n", iou_thresh);
    int n_kept = 0;
    for (int i = 0; i < N_boxes; i++) {
        if (kept[i] == 1) {
            printf("  KEPT  box %d: score=%.2f  [%.0f,%.0f,%.0f,%.0f]\n",
                   i, boxes[i*5+4],
                   boxes[i*5+0], boxes[i*5+1], boxes[i*5+2], boxes[i*5+3]);
            n_kept++;
        } else {
            printf("  SUPP  box %d: score=%.2f\n", i, boxes[i*5+4]);
        }
    }
    printf("\n");
    printf("[01] check: boxes kept = %d (expected 2)\n", n_kept);

    free(boxes); free(order); free(kept); free(scores_col);
    return 0;
}
