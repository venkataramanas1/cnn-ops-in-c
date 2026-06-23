/* ============================================================================
 * [02] Binary Cross-Entropy Loss  --  BCE for sigmoid outputs
 * ----------------------------------------------------------------------------
 * BCE is the loss for binary classification where each output is a single
 * sigmoid-activated probability p ∈ (0,1) vs a binary label y ∈ {0,1}.
 *
 * Formula per element:
 *   L = -[ y * log(p) + (1-y) * log(1-p) ]
 * Mean over all N elements.
 *
 * Models: SE block gate training, binary detection (object/no-object),
 *         VLA object-present confidence head, multi-label classification.
 *
 * Layout: predictions [N], labels [N]
 *
 * Build: gcc 02_binary_cross_entropy.c -o 02_binary_cross_entropy -lm
 * ============================================================================ */
#include <stdio.h>
#include <math.h>

/* ---- print arrays side by side ------------------------------------------- */
static void show(const char *title, const float *pred, const float *label, int N)
{
    printf("%s  [N=%d]\n", title, N);
    printf("  %10s  %10s  %10s\n", "pred(p)", "label(y)", "element loss");
    for (int i = 0; i < N; i++) {
        float p = pred[i];
        float y = label[i];
        /* clamp to avoid log(0) */
        float p_safe   = p < 1e-9f ? 1e-9f : (p > 1.0f - 1e-9f ? 1.0f - 1e-9f : p);
        float loss_i   = -(y * logf(p_safe) + (1.0f - y) * logf(1.0f - p_safe));
        printf("  %10.4f  %10.1f  %10.4f\n", p, y, loss_i);
    }
    printf("\n");
}

/* ---- BCE loss: returns mean loss, fills per_elem[N] ---------------------- */
static float bce(const float *pred, const float *label, float *per_elem, int N)
{
    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        float p = pred[i];
        float y = label[i];

        /* clamp p away from 0 and 1 to prevent log(0) = -inf */
        float p_safe = p < 1e-9f ? 1e-9f : (p > 1.0f - 1e-9f ? 1.0f - 1e-9f : p);

        /* -y * log(p): penalizes when y=1 and p is small */
        float term_pos = y * logf(p_safe);

        /* -(1-y) * log(1-p): penalizes when y=0 and p is large */
        float term_neg = (1.0f - y) * logf(1.0f - p_safe);

        per_elem[i] = -(term_pos + term_neg);  /* combine and negate */
        total += per_elem[i];
    }
    return total / (float)N;                   /* mean over batch */
}

int main(void)
{
    int N = 4;

    /* predictions from a sigmoid output */
    float pred[4]  = { 0.9f, 0.1f, 0.8f, 0.2f };

    /* binary labels */
    float label[4] = { 1.0f, 0.0f, 1.0f, 1.0f };

    show("PREDICTIONS vs LABELS (with element losses)", pred, label, N);

    float per_elem[4];
    float mean_loss = bce(pred, label, per_elem, N);

    printf("mean BCE loss = %.4f\n\n", mean_loss);

    /* checks */
    printf("[02] check: p=0.9 y=1  loss=%.4f  (expected ≈ 0.1054 = -log(0.9))\n",
           per_elem[0]);
    printf("[02] check: p=0.1 y=0  loss=%.4f  (expected ≈ 0.1054 = -log(0.9))\n",
           per_elem[1]);
    printf("[02] check: p=0.8 y=1  loss=%.4f  (expected ≈ 0.2231 = -log(0.8))\n",
           per_elem[2]);
    printf("[02] check: p=0.2 y=1  loss=%.4f  (expected ≈ 1.6094 = -log(0.2))\n\n",
           per_elem[3]);

    return 0;
}
