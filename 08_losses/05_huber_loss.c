/* ============================================================================
 * [05] Huber Loss  --  Smooth L1 / robust regression
 * ----------------------------------------------------------------------------
 * Huber: quadratic near zero (like MSE, smooth gradients), linear for large
 * errors (like L1, robust to outliers). The delta threshold is where quadratic
 * becomes linear.
 *
 * Formula (per element, delta=1.0):
 *   L(x) = 0.5 * x^2               if |x| < delta
 *         = delta * (|x| - 0.5*delta)  otherwise
 * where x = pred - target
 *
 * At delta boundary:  0.5*delta^2 = delta*(delta - 0.5*delta) = 0.5*delta^2  ✓
 * Gradient is continuous: x for quadratic, ±delta for linear (no kink).
 *
 * Models: Faster R-CNN bounding box regression, VLA end-effector position
 *         regression (robust to outlier demonstrations in imitation learning).
 *
 * Build: gcc 05_huber_loss.c -o 05_huber_loss -lm
 * ============================================================================ */
#include <stdio.h>
#include <math.h>

#define DELTA 1.0f

/* ---- Huber loss for a single error x = pred - target --------------------- */
static float huber(float x)
{
    float ax = fabsf(x);                  /* |x| */

    if (ax < DELTA) {
        return 0.5f * x * x;             /* quadratic region: smooth near 0 */
    } else {
        return DELTA * (ax - 0.5f * DELTA);  /* linear region: robust to outliers */
    }
}

/* ---- Huber gradient (for reference) -------------------------------------- */
static float huber_grad(float x)
{
    float ax = fabsf(x);
    if (ax < DELTA) {
        return x;                         /* gradient = x (like MSE) */
    } else {
        return (x > 0) ? DELTA : -DELTA; /* gradient = ±delta (bounded, like L1) */
    }
}

/* ---- show table of errors and their losses -------------------------------- */
static void show(const char *title, const float *errors, int N)
{
    printf("%s  [delta=%.1f]\n", title, DELTA);
    printf("  %8s  %12s  %12s  %12s  %6s\n",
           "error x", "huber loss", "grad", "0.5*x^2(MSE)", "region");
    for (int i = 0; i < N; i++) {
        float x  = errors[i];
        float hl = huber(x);
        float g  = huber_grad(x);
        float mse_val = 0.5f * x * x;
        printf("  %8.2f  %12.4f  %12.4f  %12.4f  %s\n",
               x, hl, g, mse_val,
               fabsf(x) < DELTA ? "quadratic" : "linear");
    }
    printf("\n");
}

int main(void)
{
    /* range of errors spanning both regions */
    float errors[] = { -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f };
    int N = 7;

    show("HUBER LOSS TABLE", errors, N);

    /* checks */
    float h_0p5 = huber(0.5f);
    float h_2p0 = huber(2.0f);
    float h_neg3 = huber(-3.0f);

    printf("[05] check: x= 0.5  huber=%.4f  (expected 0.1250 = 0.5*0.25)\n",
           h_0p5);
    printf("[05] check: x= 2.0  huber=%.4f  (expected 1.5000 = 1*(2-0.5))\n",
           h_2p0);
    printf("[05] check: x=-3.0  huber=%.4f  (expected 2.5000 = 1*(3-0.5))\n\n",
           h_neg3);

    /* show continuity at delta boundary */
    float h_just_inside  = huber(0.9999f);
    float h_just_outside = huber(1.0001f);
    printf("[05] check boundary continuity: huber(0.9999)=%.6f  huber(1.0001)=%.6f\n\n",
           h_just_inside, h_just_outside);

    return 0;
}
