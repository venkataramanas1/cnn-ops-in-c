/* ============================================================================
 * [04] Focal Loss  --  Class-imbalance robust detection loss
 * ----------------------------------------------------------------------------
 * (1-p_t)^gamma: when the model is confident (p_t→1), this factor→0,
 * suppressing the loss. Focal loss was the key to making single-stage detection
 * work on class-imbalanced anchor sets. (RetinaNet, YOLOv5/v8)
 *
 * Formula:
 *   FL = -alpha * (1-p_t)^gamma * log(p_t)
 * where:
 *   p_t = p      if y = 1  (true positive)
 *   p_t = 1 - p  if y = 0  (true negative)
 *   alpha = 0.25 (down-weights frequent background)
 *   gamma = 2.0  (focusing exponent)
 *
 * Compare: standard CE = -log(p_t)  (alpha=1, gamma=0 special case)
 *
 * Layout: [N] sigmoid predictions and binary labels
 *
 * Build: gcc 04_focal_loss.c -o 04_focal_loss -lm
 * ============================================================================ */
#include <stdio.h>
#include <math.h>

#define ALPHA  0.25f
#define GAMMA  2.0f

/* ---- focal loss for a single element -------------------------------------- */
static float focal_element(float p, float y)
{
    /* clamp to avoid log(0) */
    float p_safe = p < 1e-9f ? 1e-9f : (p > 1.0f - 1e-9f ? 1.0f - 1e-9f : p);

    /* p_t: confidence for the ground-truth class */
    float p_t = (y > 0.5f) ? p_safe : (1.0f - p_safe);

    /* (1-p_t)^gamma: the focusing factor — near 0 for easy, near 1 for hard */
    float focus = powf(1.0f - p_t, GAMMA);

    /* standard CE part: -log(p_t) */
    float ce = -logf(p_t);

    /* focal loss: scale CE by alpha and focusing factor */
    return ALPHA * focus * ce;
}

/* ---- standard CE for comparison ------------------------------------------ */
static float ce_element(float p, float y)
{
    float p_safe = p < 1e-9f ? 1e-9f : (p > 1.0f - 1e-9f ? 1.0f - 1e-9f : p);
    float p_t    = (y > 0.5f) ? p_safe : (1.0f - p_safe);
    return -logf(p_t);                    /* raw CE without alpha/gamma */
}

/* ---- print table of examples --------------------------------------------- */
static void show(const char *label, const float *p, const float *y,
                 const float *fl, const float *ce, int N)
{
    printf("%s  [N=%d]\n", label, N);
    printf("  %-20s  %6s  %6s  %10s  %10s  %s\n",
           "description", "pred", "label", "focal_loss", "CE_loss", "focal<CE?");
    for (int i = 0; i < N; i++) {
        const char *desc[] = {
            "easy-positive (p=0.95,y=1)",
            "hard-positive (p=0.40,y=1)",
            "easy-negative (p=0.05,y=0)",
            "hard-negative (p=0.60,y=0)"
        };
        printf("  %-28s  %6.2f  %6.1f  %10.4f  %10.4f  %s\n",
               desc[i], p[i], y[i], fl[i], ce[i],
               fl[i] < ce[i] ? "YES" : "NO");
    }
    printf("\n");
}

int main(void)
{
    int N = 4;

    /* four representative anchor examples */
    float pred[4]  = { 0.95f, 0.40f, 0.05f, 0.60f };
    float label[4] = { 1.0f,  1.0f,  0.0f,  0.0f  };

    float fl[4], ce[4];
    for (int i = 0; i < N; i++) {
        fl[i] = focal_element(pred[i], label[i]);
        ce[i] = ce_element(pred[i],   label[i]);
    }

    show("FOCAL LOSS vs CE  (alpha=0.25, gamma=2.0)", pred, label, fl, ce, N);

    /* checks */
    printf("[04] check: easy-positive FL=%.4f  CE=%.4f  (FL < CE: %s)\n",
           fl[0], ce[0], fl[0] < ce[0] ? "PASS" : "FAIL");
    printf("[04] check: hard-positive FL=%.4f  CE=%.4f  (FL < CE: %s)\n",
           fl[1], ce[1], fl[1] < ce[1] ? "PASS" : "FAIL");
    printf("[04] check: easy-negative FL=%.4f  CE=%.4f  (FL < CE: %s)\n",
           fl[2], ce[2], fl[2] < ce[2] ? "PASS" : "FAIL");
    printf("[04] check: hard-negative FL=%.4f  CE=%.4f  (FL < CE: %s)\n\n",
           fl[3], ce[3], fl[3] < ce[3] ? "PASS" : "FAIL");

    /* also verify focusing factor explicitly for easy-positive */
    float p_t_easy = pred[0];                         /* y=1, so p_t = p */
    float focus    = powf(1.0f - p_t_easy, GAMMA);   /* (1-0.95)^2 = 0.0025 */
    printf("[04] check: focus factor for easy-positive = %.6f (expected 0.0025)\n\n",
           focus);

    return 0;
}
