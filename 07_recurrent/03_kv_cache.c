/*
 * 03_kv_cache.c
 *
 * Op: KV cache for autoregressive attention — the key memory optimization
 *     in all LLM-based VLA inference.
 *
 * KV cache: without it, generating token t requires recomputing K,V for ALL
 * previous tokens — O(t^2) cost. With cache, each new token only computes ONE
 * new K,V row. Used in RT-2/OpenVLA/pi0 inference to generate action tokens
 * autoregressively.
 *
 * STATIC shape: max_seq fixed at compile time (common on edge, avoids malloc)
 * DYNAMIC shape: cache grows with T; requires realloc or pre-allocated max_len buffer
 * Edge NPUs: often require static max_seq declared in ONNX model
 *
 * 4D tensor layout [N][T_cached][H][Dh]:
 *   flat: n*T_max*H*Dh + t*H*Dh + h_head*Dh + d
 *   T_cached grows dynamically each step — this IS the dynamic shape.
 *   T_max is the pre-allocated maximum (static on edge devices).
 *
 * Demo scenario:
 *   Dh=4, max_seq=5, N=1, H=1 (single head for clarity)
 *   Pre-fill: encode 3 context tokens (t=0,1,2) → K_cache, V_cache[0..2]
 *   Generation: steps t=3 and t=4, each appends one new K/V row,
 *               then Q attends over the full growing cache.
 *
 * Attention:
 *   scores[t] = Q[0..Dh-1] . K_cache[t][0..Dh-1] / sqrt(Dh)
 *   weights  = softmax(scores[0..T_cached-1])
 *   output   = sum_t weights[t] * V_cache[t][0..Dh-1]
 *
 * WHERE IN MODELS:
 *   - RT-2, OpenVLA, pi0: all generate action tokens autoregressively.
 *   - Without KV cache, decoding a 16-token action sequence would recompute
 *     every past key/value at every new token — prohibitive on edge hardware.
 *
 * Build:
 *   gcc -O2 -o 03_kv_cache 03_kv_cache.c -lm && ./03_kv_cache
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Static shapes for demo (edge NPU friendly — no malloc)              */
/* ------------------------------------------------------------------ */

#define N       1    /* batch size (single sequence for clarity) */
#define H       1    /* number of attention heads                 */
#define Dh      4    /* head dimension                            */
#define T_MAX   5    /* pre-allocated max sequence length         */
                     /* STATIC: declared at compile time for edge */

/* Flat index helpers for [N][T][H][Dh] layout */
/* flat: n*T_max*H*Dh + t*H*Dh + h_head*Dh + d  */
#define IDX4(n, t, h_head, d)  \
    ((n) * T_MAX * H * Dh + (t) * H * Dh + (h_head) * Dh + (d))

/* ------------------------------------------------------------------ */
/* softmax in-place over an array of length len                        */
/* ------------------------------------------------------------------ */
static void softmax_inplace(float *x, int len)
{
    int i;
    float max_val = x[0];
    float sum = 0.0f;

    /* find max for numerical stability */
    for (i = 1; i < len; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }

    /* exponentiate and accumulate */
    for (i = 0; i < len; i++)
    {
        x[i] = expf(x[i] - max_val);  /* subtract max before exp for stability */
        sum += x[i];
    }

    /* normalize to sum to 1 */
    for (i = 0; i < len; i++)
    {
        x[i] /= sum;
    }
}

/* ------------------------------------------------------------------ */
/* attend: Q[1][Dh] attends over K_cache[0..T_cached-1][Dh]           */
/*         output accumulated from V_cache weighted by attention       */
/*                                                                      */
/* Dynamic shape: T_cached grows each generation step —                */
/*   int stride_t_cache = H * Dh;  step along time axis in cache      */
/* ------------------------------------------------------------------ */
static void attend(
    const float *Q,         /* [Dh] — query for current token           */
    const float *K_cache,   /* [N][T_max][H][Dh] — key cache            */
    const float *V_cache,   /* [N][T_max][H][Dh] — value cache          */
    float       *out,       /* [Dh] — attention output                  */
    float       *weights,   /* [T_cached] — attention weights (output)  */
    int          t_cached)  /* number of cached tokens (grows each step) */
{
    int t, d;
    float scale = 1.0f / sqrtf((float)Dh);  /* 1/sqrt(Dh) scaling factor */

    /*
     * T_cached is dynamic: grows by 1 each generation step.
     * stride_t_cache = H * Dh — step along time axis in the cache.
     */
    int stride_t_cache = H * Dh;  /* step along time axis in cache */

    /* Step 1: compute raw attention scores Q . K_cache[t] / sqrt(Dh) */
    float scores[T_MAX];
    for (t = 0; t < t_cached; t++)
    {
        float dot = 0.0f;
        for (d = 0; d < Dh; d++)
        {
            /* K_cache[n=0][t][h=0][d]: flat IDX4(0, t, 0, d) */
            dot += Q[d] * K_cache[IDX4(0, t, 0, d)];
        }
        scores[t] = dot * scale;  /* scale by 1/sqrt(Dh) before softmax */
    }

    /* Step 2: softmax over scores to get attention weights */
    memcpy(weights, scores, t_cached * sizeof(float));
    softmax_inplace(weights, t_cached);  /* weights sum to 1.0 */

    /* Step 3: weighted sum of V_cache */
    for (d = 0; d < Dh; d++)
    {
        out[d] = 0.0f;
    }
    for (t = 0; t < t_cached; t++)
    {
        for (d = 0; d < Dh; d++)
        {
            /* V_cache[n=0][t][h=0][d]: flat IDX4(0, t, 0, d) */
            out[d] += weights[t] * V_cache[IDX4(0, t, 0, d)];
        }
    }
}

/* ------------------------------------------------------------------ */
/* print_cache: show current K or V cache contents up to t_cached      */
/* ------------------------------------------------------------------ */
static void print_cache(const char *name, const float *cache, int t_cached)
{
    int t, d;
    printf("  %s_cache[0..%d]:\n", name, t_cached - 1);
    for (t = 0; t < t_cached; t++)
    {
        printf("    t=%d: [", t);
        for (d = 0; d < Dh; d++)
        {
            /* cache[n=0][t][h=0][d] */
            printf(" %5.2f", cache[IDX4(0, t, 0, d)]);
        }
        printf(" ]\n");
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    int t, d;

    printf("=== 03_kv_cache ===\n");
    printf("KV cache: each new token appends ONE K,V row instead of\n");
    printf("recomputing all past tokens. T_cached grows dynamically.\n");
    printf("STATIC max_seq=%d pre-allocated (no malloc, edge-friendly).\n\n",
           T_MAX);

    printf("Config: N=%d H=%d Dh=%d T_MAX=%d\n\n", N, H, Dh, T_MAX);

    /*
     * T_cached is dynamic: sequence length not known at compile time.
     * It grows from 0 → T_MAX as tokens are encoded/generated.
     * int stride_t_cache = H * Dh — step along time axis in cache.
     */
    int t_cached = 0;  /* how many tokens are in the cache so far */

    /* Pre-allocated KV cache: [N][T_MAX][H][Dh] — static max_seq on edge */
    float K_cache[N * T_MAX * H * Dh];
    float V_cache[N * T_MAX * H * Dh];
    memset(K_cache, 0, sizeof(K_cache));
    memset(V_cache, 0, sizeof(V_cache));

    /*
     * Pre-fill: 3 context tokens (t=0,1,2).
     * In a real VLA: these are the image/instruction prefix tokens.
     * Keys and values are distinct to make attention scores interesting.
     *
     * K[t=0] = [1, 0, 0, 0]   V[t=0] = [0.1, 0.2, 0.3, 0.4]
     * K[t=1] = [0, 1, 0, 0]   V[t=1] = [0.5, 0.6, 0.7, 0.8]
     * K[t=2] = [0, 0, 1, 0]   V[t=2] = [0.9, 1.0, 1.1, 1.2]
     */
    printf("=== PRE-FILL: encoding 3 context tokens ===\n");

    /* context token key/value definitions */
    float ctx_K[3][Dh] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f}
    };
    float ctx_V[3][Dh] = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.5f, 0.6f, 0.7f, 0.8f},
        {0.9f, 1.0f, 1.1f, 1.2f}
    };

    for (t = 0; t < 3; t++)
    {
        /* append K and V for context token t into the cache */
        for (d = 0; d < Dh; d++)
        {
            K_cache[IDX4(0, t_cached, 0, d)] = ctx_K[t][d];
            V_cache[IDX4(0, t_cached, 0, d)] = ctx_V[t][d];
        }
        t_cached++;  /* cache grows: T_cached is dynamic */
        printf("  appended context token %d → T_cached=%d\n", t, t_cached);
    }

    printf("\nCache after pre-fill (T_cached=%d):\n", t_cached);
    print_cache("K", K_cache, t_cached);
    print_cache("V", V_cache, t_cached);

    /*
     * Generation steps t=3 and t=4.
     * Each step:
     *   1. Compute Q for the new token (small synthetic queries).
     *   2. Compute new K, V for the new token and append to cache.
     *   3. Run attention: Q attends over ALL cached K[0..t_cached-1].
     *   4. Collect output.
     *
     * T_cached grows by 1 each step — this is the dynamic shape in action.
     */

    /* synthetic Q, K, V for generation tokens */
    float gen_Q[2][Dh] = {
        {0.5f, 0.5f, 0.0f, 0.0f},   /* gen step 3: query focused on t=0,t=1 */
        {0.0f, 0.0f, 1.0f, 1.0f}    /* gen step 4: query focused on t=2 */
    };
    float gen_K[2][Dh] = {
        {0.0f, 0.0f, 0.0f, 1.0f},   /* new key for step 3 */
        {1.0f, 1.0f, 0.0f, 0.0f}    /* new key for step 4 */
    };
    float gen_V[2][Dh] = {
        {2.0f, 2.1f, 2.2f, 2.3f},   /* new value for step 3 */
        {3.0f, 3.1f, 3.2f, 3.3f}    /* new value for step 4 */
    };

    float out[Dh];
    float weights[T_MAX];
    float weight_sum;
    int gen_idx;

    for (gen_idx = 0; gen_idx < 2; gen_idx++)
    {
        int step = 3 + gen_idx;  /* global sequence position */
        printf("\n=== GENERATION STEP t=%d ===\n", step);

        /* Step 1: append the new token's K, V to the cache */
        for (d = 0; d < Dh; d++)
        {
            K_cache[IDX4(0, t_cached, 0, d)] = gen_K[gen_idx][d];
            V_cache[IDX4(0, t_cached, 0, d)] = gen_V[gen_idx][d];
        }
        t_cached++;  /* T_cached grows: one new K,V row added, NOT recomputed */

        printf("  Appended new K/V → T_cached now %d (only 1 new row, not %d recomputations)\n",
               t_cached, t_cached);

        printf("\nCache after append (T_cached=%d):\n", t_cached);
        print_cache("K", K_cache, t_cached);
        print_cache("V", V_cache, t_cached);

        /* Step 2: run attention — Q attends over the full grown cache */
        printf("\n  Q for step %d: [", step);
        for (d = 0; d < Dh; d++)
        {
            printf(" %5.2f", gen_Q[gen_idx][d]);
        }
        printf(" ]\n");

        attend(gen_Q[gen_idx], K_cache, V_cache, out, weights, t_cached);

        /* print attention scores and weights */
        printf("\n  Attention scores (Q.K^T / sqrt(%d)) and weights:\n", Dh);
        for (t = 0; t < t_cached; t++)
        {
            printf("    t=%d: weight=%7.4f\n", t, weights[t]);
        }

        /* verify weights sum to 1.0 */
        weight_sum = 0.0f;
        for (t = 0; t < t_cached; t++)
        {
            weight_sum += weights[t];
        }
        printf("  Weight sum = %.6f  %s\n",
               weight_sum,
               (weight_sum > 0.9999f && weight_sum < 1.0001f) ? "(sum=1.0 OK)" : "SUM != 1.0!");

        printf("\n  Output = sum_t weight[t]*V[t]: [");
        for (d = 0; d < Dh; d++)
        {
            printf(" %6.3f", out[d]);
        }
        printf(" ]\n");
    }

    /* Final check: weights sum to 1.0 at last generation step */
    weight_sum = 0.0f;
    for (t = 0; t < t_cached; t++)
    {
        weight_sum += weights[t];
    }
    int sum_ok = (weight_sum > 0.9999f && weight_sum < 1.0001f);

    printf("\n=== FINAL CHECK ===\n");
    printf("Attention weights sum at last step: %.6f\n", weight_sum);
    printf("RESULT: %s\n", sum_ok ? "PASS" : "FAIL");

    printf("\nSummary: T grew from 0 to %d dynamically.\n", t_cached);
    printf("Each generation step appended 1 K/V row (O(T) total work)\n");
    printf("vs. O(T^2) if recomputing all past keys each step.\n");

    return 0;
}
