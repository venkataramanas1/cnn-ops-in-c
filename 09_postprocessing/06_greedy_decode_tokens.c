/* ============================================================================
 * [06] Autoregressive greedy decode  --  VLA action token generation
 * ----------------------------------------------------------------------------
 * Autoregressive greedy decode: at each step, the model sees its own previous
 * output. The sequence grows dynamically -- this is why static shapes require a
 * max_len cap in ONNX/TFLite export.
 *
 *   At each step t:
 *     1. logits[N][1][vocab_size]  <- model forward pass (simulated here)
 *     2. probs  = softmax(logits)
 *     3. token  = argmax(probs)     <- greedy decode (pick most likely token)
 *     4. append token to sequence
 *     5. if token == EOS_ID: stop
 *
 *   Output sequence length T is FULLY dynamic: not known until EOS is emitted.
 *   A max_len cap is used as a safety guard.
 *
 * DYNAMIC SHAPES:
 *   - logits at each step: [N][1][vocab_size]
 *   - growing sequence: [N][dynamic_T] -- realloc'd as tokens are appended
 *
 * MODELS: RT-2, OpenVLA, pi0 -- ALL LLM-based VLA models generate action tokens
 *   autoregressively. The number of action tokens is a dynamic shape that depends
 *   on the task. Beam search (top-K) is the non-greedy variant.
 *
 * Demo: vocab_size=6, max_len=5, N=1.
 *   Pre-defined logit matrices simulate 4 decode steps spelling out [2,3,1,5].
 *   EOS_ID=5; show decode at each step; stop when EOS generated.
 *   Check: sequence terminates at EOS and length is correct.
 *
 * Build: gcc 06_greedy_decode_tokens.c -o 06_greedy_decode_tokens -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- softmax in-place for one logit vector of length V --- */
static void softmax(float *v, int V)
{
    float maxv = v[0];
    for (int i = 1; i < V; i++) {
        if (v[i] > maxv) { maxv = v[i]; }  /* find max for numerical stability */
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        v[i] = expf(v[i] - maxv);          /* stable exp: subtract max before exp */
        sum += v[i];                         /* accumulate partition function */
    }
    for (int i = 0; i < V; i++) {
        v[i] /= sum;                         /* normalize to valid probability distribution */
    }
}

int main(void)
{
    /* runtime shape variables */
    int N          = 1;   /* batch size */
    int vocab_size = 6;   /* vocabulary size -- runtime variable */
    int max_len    = 5;   /* maximum sequence length cap (for ONNX export safety) */
    int EOS_ID     = 5;   /* end-of-sequence token ID */

    /* --- pre-defined logit matrices for 4 simulated decode steps ---
       Each row is [N=1][1][vocab_size] at one time step.
       The high logit position determines the greedy token.
       Step 0: high at index 2 -> token 2
       Step 1: high at index 3 -> token 3
       Step 2: high at index 1 -> token 1
       Step 3: high at index 5 -> token 5 (EOS) -> STOP */
    int   max_steps = 4;  /* number of simulated forward passes */
    float sim_logits[4][6] = {
        {0.1f, 0.2f, 5.0f, 0.3f, 0.1f, 0.0f},  /* step 0: argmax = 2 */
        {0.2f, 0.1f, 0.3f, 6.0f, 0.1f, 0.0f},  /* step 1: argmax = 3 */
        {0.1f, 4.5f, 0.2f, 0.3f, 0.1f, 0.0f},  /* step 2: argmax = 1 */
        {0.0f, 0.1f, 0.1f, 0.1f, 0.0f, 7.0f},  /* step 3: argmax = 5 (EOS) */
    };

    /* heap-allocate logit workspace: [N][1][vocab_size] */
    float *logits = (float *)malloc((size_t)N * 1 * vocab_size * sizeof(float));

    /* dynamic sequence: start with capacity max_len, grow as needed
       sequence[n][t] -- for N=1 this is just a flat array of token IDs */
    int  seq_len = 0;                                        /* current dynamic length */
    int *sequence = (int *)malloc((size_t)max_len * sizeof(int));  /* dynamic sequence buffer */

    printf("[06] Autoregressive greedy decode\n");
    printf("     vocab_size=%d  EOS_ID=%d  max_len=%d\n\n", vocab_size, EOS_ID, max_len);

    /* --- autoregressive decode loop ---
       Each iteration is one model forward pass (simulated).
       The loop terminates when EOS is generated or max_len is reached. */
    int step = 0;
    while (step < max_len) {                        /* max_len cap prevents infinite loop */

        /* --- simulate model forward pass: copy pre-defined logits for this step ---
           In a real VLA model this would be: logits = model(sequence_so_far).
           Logit tensor is [N=1][1][vocab_size]; stride = 0*1*V + 0*V + c = c */
        for (int c = 0; c < vocab_size; c++) {
            logits[c] = (step < max_steps) ? sim_logits[step][c] : 0.0f;
        }

        /* print raw logits for this step */
        printf("  step %d  logits: [", step);
        for (int c = 0; c < vocab_size; c++) {
            printf("%5.1f%s", logits[c], c < vocab_size-1 ? ", " : "");
        }
        printf("]\n");

        /* --- softmax over vocab ---
           Converts raw logits to a probability distribution over vocab tokens. */
        softmax(logits, vocab_size);  /* in-place; logits now holds probs */

        /* --- greedy decode: argmax over vocab probabilities ---
           Greedy picks the single most likely token at each step.
           This is equivalent to temperature T -> 0 in the limit. */
        int   best_token = 0;
        float best_prob  = logits[0];
        for (int c = 1; c < vocab_size; c++) {
            if (logits[c] > best_prob) {
                best_prob  = logits[c];  /* update running maximum probability */
                best_token = c;          /* track index of the winning token */
            }
        }

        /* print probs and selected token */
        printf("         probs:  [");
        for (int c = 0; c < vocab_size; c++) {
            printf("%5.3f%s", logits[c], c < vocab_size-1 ? ", " : "");
        }
        printf("]\n");
        printf("         -> greedy token = %d  (prob=%.3f)", best_token, best_prob);

        /* --- append token to dynamic sequence ---
           In a real system, sequence grows by realloc if needed.
           Here max_len is our pre-allocated cap. */
        sequence[seq_len] = best_token;  /* append to [N][dynamic_T] sequence */
        seq_len++;                        /* sequence length grows by 1 each step */

        /* --- check EOS: stop generation if end-of-sequence token was produced --- */
        if (best_token == EOS_ID) {
            printf("  <- EOS generated, STOP\n\n");
            break;  /* dynamic sequence length is now fixed */
        }
        printf("\n");

        step++;
    }

    /* --- print final generated sequence --- */
    printf("[06] Generated sequence (dynamic length T=%d): [", seq_len);
    for (int t = 0; t < seq_len; t++) {
        printf(" %d", sequence[t]);
    }
    printf(" ]\n\n");

    /* --- checks --- */
    /* expected: tokens [2, 3, 1, 5] where 5 is EOS */
    int expected_seq[] = {2, 3, 1, 5};
    int expected_len   = 4;
    int seq_ok = (seq_len == expected_len);
    if (seq_ok) {
        for (int t = 0; t < seq_len; t++) {
            if (sequence[t] != expected_seq[t]) { seq_ok = 0; }
        }
    }
    printf("[06] check: sequence matches expected [2,3,1,5] -> %s\n", seq_ok ? "YES" : "NO");
    printf("[06] check: last token is EOS (%d) -> %s\n",
           EOS_ID,
           (seq_len > 0 && sequence[seq_len-1] == EOS_ID) ? "YES" : "NO");
    printf("[06] check: dynamic seq_len=%d < max_len=%d -> %s\n",
           seq_len, max_len, (seq_len < max_len) ? "YES (stopped early at EOS)" : "NO");

    free(logits); free(sequence);
    return 0;
}
