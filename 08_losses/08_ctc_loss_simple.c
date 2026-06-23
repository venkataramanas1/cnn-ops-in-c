/* ============================================================================
 * [08] CTC Loss (Simplified)  --  Greedy decode of sequence predictions
 * ----------------------------------------------------------------------------
 * CTC allows the model to output at every frame without knowing alignment.
 * Greedy decode: take argmax at each step, collapse consecutive duplicates,
 * remove blank. The full CTC loss uses forward-backward DP — here we show
 * the greedy decode used at inference.
 *
 * Models: speech recognition (Wav2Vec2, Whisper CTC head), OCR,
 *         some VLA language output decoders.
 *
 * Input: [T][C] per-frame class log-probabilities (or raw logits for greedy)
 *   T = number of frames (runtime)
 *   C = number of classes including blank (blank = C-1 by convention)
 *
 * Greedy decode algorithm:
 *   1. argmax at each frame → raw label sequence of length T
 *   2. collapse: remove consecutive duplicates  e.g. [0,0,1,1] → [0,1]
 *   3. remove blank tokens                      e.g. [0,3,1,3] → [0,1]
 *
 * Stride: element [t][c] = base + t*C + c
 *
 * Build: gcc 08_ctc_loss_simple.c -o 08_ctc_loss_simple -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

/* ---- print per-frame argmax sequence ------------------------------------- */
static void show_frames(const char *title, const int *seq, int T, int blank)
{
    printf("%s  [T=%d  blank=%d]\n", title, T, blank);
    printf("  [");
    for (int t = 0; t < T; t++) {
        if (seq[t] == blank) {
            printf(" BLK");
        } else {
            printf("  %2d", seq[t]);
        }
        if (t < T - 1) { printf(","); }
    }
    printf(" ]\n\n");
}

/* ---- print decoded sequence ---------------------------------------------- */
static void show_decoded(const char *title, const int *seq, int len)
{
    printf("%s  [len=%d]\n", title, len);
    printf("  [");
    for (int i = 0; i < len; i++) {
        printf("  %d%s", seq[i], i < len - 1 ? "," : "");
    }
    printf(" ]\n\n");
}

/* ---- step 1: argmax at each frame ---------------------------------------- */
/* fills argmax_seq[T]; returns nothing */
static void frame_argmax(const float *logits, int *argmax_seq, int T, int C)
{
    int stride_t = C;    /* [t][c] = logits + t*C + c */

    for (int t = 0; t < T; t++) {
        int   best_c   = 0;
        float best_val = logits[t * stride_t + 0];

        for (int c = 1; c < C; c++) {
            float val = logits[t * stride_t + c];   /* logit for class c at frame t */
            if (val > best_val) {
                best_val = val;
                best_c   = c;
            }
        }
        argmax_seq[t] = best_c;   /* most likely class at this frame */
    }
}

/* ---- step 2: collapse consecutive duplicates ----------------------------- */
/* fills collapsed[T]; returns collapsed length */
static int collapse_repeats(const int *seq, int *collapsed, int T)
{
    int len = 0;
    for (int t = 0; t < T; t++) {
        /* only keep if different from the previous kept token */
        if (len == 0 || seq[t] != collapsed[len - 1]) {
            collapsed[len] = seq[t];
            len++;
        }
    }
    return len;
}

/* ---- step 3: remove blank tokens ----------------------------------------- */
/* fills decoded[T]; returns decoded length */
static int remove_blanks(const int *seq, int *decoded, int len, int blank)
{
    int out_len = 0;
    for (int i = 0; i < len; i++) {
        if (seq[i] != blank) {
            decoded[out_len] = seq[i];
            out_len++;
        }
    }
    return out_len;
}

int main(void)
{
    /* runtime parameters */
    int T = 8;   /* number of frames */
    int C = 4;   /* classes: 0, 1, 2, and blank=3 */
    int blank = C - 1;   /* blank token is last class by convention */

    /* [T][C] logits — rows are frames, columns are class scores
     * we construct these so argmax yields: [0, 0, BLK, 1, BLK, 2, 2, BLK]
     * collapse repeats:  [0, BLK, 1, BLK, 2, BLK]
     * remove blanks:     [0, 1, 2]  (three distinct characters) */
    float logits[8 * 4] = {
        /* t=0: class 0 wins */  5.0f, 1.0f, 1.0f, 0.0f,
        /* t=1: class 0 wins */  5.0f, 1.0f, 1.0f, 0.0f,
        /* t=2: blank wins   */  0.0f, 0.0f, 0.0f, 5.0f,
        /* t=3: class 1 wins */  1.0f, 5.0f, 1.0f, 0.0f,
        /* t=4: blank wins   */  0.0f, 0.0f, 0.0f, 5.0f,
        /* t=5: class 2 wins */  1.0f, 1.0f, 5.0f, 0.0f,
        /* t=6: class 2 wins */  1.0f, 1.0f, 5.0f, 0.0f,
        /* t=7: blank wins   */  0.0f, 0.0f, 0.0f, 5.0f
    };

    printf("CTC GREEDY DECODE  [T=%d  C=%d  blank=%d]\n\n", T, C, blank);

    /* ---- step 1: argmax at each frame ------------------------------------ */
    int *argmax_seq = malloc(T * sizeof(int));
    frame_argmax(logits, argmax_seq, T, C);
    show_frames("Step 1: per-frame argmax", argmax_seq, T, blank);

    /* ---- step 2: collapse consecutive duplicates ------------------------- */
    int *collapsed = malloc(T * sizeof(int));
    int collapsed_len = collapse_repeats(argmax_seq, collapsed, T);
    show_frames("Step 2: after collapsing repeats", collapsed, collapsed_len, blank);
    printf("  collapsed length = %d  (was T=%d)\n\n", collapsed_len, T);

    /* ---- step 3: remove blank tokens ------------------------------------- */
    int *decoded = malloc(T * sizeof(int));
    int decoded_len = remove_blanks(collapsed, decoded, collapsed_len, blank);
    show_decoded("Step 3: final decoded sequence (blanks removed)", decoded, decoded_len);

    /* checks */
    printf("[08] check: raw argmax length=%d  decoded length=%d\n", T, decoded_len);
    printf("[08] check: compression happened = %s  (decoded < T: %d < %d)\n",
           decoded_len < T ? "YES" : "NO", decoded_len, T);
    printf("[08] check: decoded = [");
    for (int i = 0; i < decoded_len; i++) {
        printf("%d%s", decoded[i], i < decoded_len - 1 ? ", " : "");
    }
    printf("]  (expected [0, 1, 2])\n\n");

    free(argmax_seq);
    free(collapsed);
    free(decoded);
    return 0;
}
