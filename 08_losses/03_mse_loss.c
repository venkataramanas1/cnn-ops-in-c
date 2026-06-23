/* ============================================================================
 * [03] Mean Squared Error Loss  --  L2 regression
 * ----------------------------------------------------------------------------
 * MSE is the L2 regression loss for continuous outputs — robot joint angles,
 * EE position, gripper state in VLA action heads (RT-2, ACT policy).
 *
 * Formula:
 *   L = mean over all elements of (pred - target)^2
 *
 * Layout: [N][T][D_action]
 *   N = batch size (runtime)
 *   T = timesteps in action sequence (runtime)
 *   D = action dimension (e.g., 7 = 6-DOF + gripper)
 *
 * Stride computation:
 *   element [n][t][d] = base + n*(T*D) + t*D + d
 *   stride_n = T*D,  stride_t = D
 *
 * Build: gcc 03_mse_loss.c -o 03_mse_loss -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---- print [N][T][D] tensor ----------------------------------------------- */
static void show(const char *title, const float *m, int N, int T, int D)
{
    int stride_n = T * D;    /* floats to skip per sample */
    int stride_t = D;        /* floats to skip per timestep */

    printf("%s  [N=%d T=%d D=%d]  stride_n=%d stride_t=%d\n",
           title, N, T, D, stride_n, stride_t);
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            printf("  [n=%d t=%d]: ", n, t);
            for (int d = 0; d < D; d++) {
                /* flat index using strides */
                printf("%7.3f", m[n * stride_n + t * stride_t + d]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

/* ---- MSE loss over [N][T][D] ---------------------------------------------- */
/* returns scalar mean loss; also fills per_sample[N] as mean over T*D */
static float mse_loss(const float *pred, const float *target,
                      float *per_sample, int N, int T, int D)
{
    int stride_n = T * D;
    int stride_t = D;

    float total = 0.0f;

    for (int n = 0; n < N; n++) {
        float sum_n = 0.0f;

        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                /* flat index [n][t][d] */
                int idx = n * stride_n + t * stride_t + d;

                float diff = pred[idx] - target[idx];  /* prediction error */
                float sq   = diff * diff;              /* squared error */
                sum_n += sq;
            }
        }

        per_sample[n] = sum_n / (float)(T * D);  /* mean over T*D elements */
        total += per_sample[n];
    }

    return total / (float)N;    /* mean over N samples */
}

int main(void)
{
    /* runtime parameters */
    int N = 2;   /* batch size */
    int T = 3;   /* action sequence length */
    int D = 2;   /* action dim: [joint_angle, gripper_state] (simplified) */

    int total_elems = N * T * D;

    float *pred   = malloc(total_elems * sizeof(float));
    float *target = malloc(total_elems * sizeof(float));

    /* target: simulated robot joint trajectory */
    float target_vals[2 * 3 * 2] = {
        /* n=0, t=0 */ 0.1f, 0.5f,
        /* n=0, t=1 */ 0.2f, 0.5f,
        /* n=0, t=2 */ 0.3f, 0.5f,
        /* n=1, t=0 */ 1.0f, 0.0f,
        /* n=1, t=1 */ 1.1f, 0.0f,
        /* n=1, t=2 */ 1.2f, 0.0f
    };
    for (int i = 0; i < total_elems; i++) {
        target[i] = target_vals[i];
    }

    /* sample 0: perfect prediction (pred == target) */
    /* sample 1: noisy prediction (add offsets)      */
    float pred_vals[2 * 3 * 2] = {
        /* n=0: perfect */
        0.1f, 0.5f,
        0.2f, 0.5f,
        0.3f, 0.5f,
        /* n=1: noisy — errors of 0.5 on each dim */
        1.5f, 0.5f,
        1.6f, 0.5f,
        1.7f, 0.5f
    };
    for (int i = 0; i < total_elems; i++) {
        pred[i] = pred_vals[i];
    }

    show("TARGET  [N][T][D]", target, N, T, D);
    show("PRED    [N][T][D]", pred,   N, T, D);

    float per_sample[2];
    float mean_loss = mse_loss(pred, target, per_sample, N, T, D);

    printf("per-sample MSE: sample0=%.4f  sample1=%.4f\n",
           per_sample[0], per_sample[1]);
    printf("mean MSE loss  = %.4f\n\n", mean_loss);

    /* checks */
    printf("[03] check: perfect pred MSE = %.4f  (expected 0.0)\n",
           per_sample[0]);
    printf("[03] check: noisy pred  MSE  = %.4f  (expected 0.25 = 0.5^2)\n\n",
           per_sample[1]);

    free(pred);
    free(target);
    return 0;
}
