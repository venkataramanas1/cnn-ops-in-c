/*
 * 15_gather_token.c
 *
 * WHAT CHANGES:  Sequence length shrinks (or reorders) — select a subset of tokens.
 *                output[i] = tokens[ indices[i] ]  for each index i.
 * WHAT STAYS:    Token feature dimension; all selected values unchanged.
 *
 * WHERE IN MODELS:
 *   - VLA action token extraction (OpenVLA, RT-2):
 *     The language model produces a full output sequence of T tokens.
 *     Only the last A tokens are action tokens (one per action dimension).
 *     indices = [T-A, T-A+1, ..., T-1] selects exactly those.
 *     In OpenVLA (A=7 for 7-DOF arm): gather the last 7 tokens.
 *   - Transformer embedding lookup: the embedding table is a [vocab][dim]
 *     matrix; a Gather with input_id indices retrieves one row per token —
 *     every forward pass starts with this op.
 *   - KV cache attention masking: gather cached key/value entries by
 *     position index to handle variable-length prefix sequences.
 *   - Sparse attention: gather only the top-k attended tokens from the
 *     full sequence (e.g. Longformer global token selection).
 *
 * DEMO:
 *   6 tokens of feature dim 4.
 *   indices = [0, 2, 4] → select tokens 0, 2, 4 (every other token).
 *   Output: 3 selected tokens.
 *
 * Build:
 *   gcc -O2 -o 15_gather_token 15_gather_token.c && ./15_gather_token
 */

#include <stdio.h>

#define T_IN  6   /* input sequence length */
#define D     4   /* feature dimension per token */
#define T_OUT 3   /* number of indices to gather */

static void show_seq(const float *t, int nt, const char *label)
{
    int i, d;
    printf("%s [T=%d][D=%d]:\n", label, nt, D);
    for (i = 0; i < nt; i++)
    {
        printf("  token %d: [", i);
        for (d = 0; d < D; d++)
        {
            printf(" %5.1f", t[i * D + d]);
        }
        printf(" ]\n");
    }
}

/*
 * gather_tokens — index-select from a sequence.
 *
 * For each output position i:
 *   src_tok = indices[i]           — which input token to copy
 *   out[i][d] = in[src_tok][d]     — copy all D features of that token
 *
 * WHY the formula is trivial:
 *   Each token is a contiguous block of D floats starting at offset
 *   tok*D.  Gather just decides WHICH block to copy by looking up
 *   indices[i] — it is a level of indirection over the token axis.
 *
 * WHY gather matters for VLA:
 *   The language model doesn't "know" which of its T output positions hold
 *   action tokens — that is encoded in the indices array (a hyperparameter
 *   set at model design time).  Gather applies this selection at inference.
 *   In ONNX: Gather node with axis=0.
 *   In PyTorch: tokens[indices]  or  torch.index_select(tokens, 0, indices).
 */
static void gather_tokens(const float *in, const int *indices, float *out, int n_out)
{
    int i, d;
    for (i = 0; i < n_out; i++)
    {
        int src_tok = indices[i]; /* which input token to copy */
        for (d = 0; d < D; d++)
        {
            /* Copy all D features from input token src_tok to output token i */
            out[i * D + d] = in[src_tok * D + d];
        }
    }
}

int main(void)
{
    float tokens[T_IN * D];
    float out[T_OUT * D];
    /* Gather every other token */
    int indices[T_OUT] = {0, 2, 4};
    int t, d;

    /* Token t: feature d gets value t*10 + d (encodes origin clearly) */
    for (t = 0; t < T_IN; t++)
    {
        for (d = 0; d < D; d++)
        {
            tokens[t * D + d] = (float)(t * 10 + d);
        }
    }

    printf("=== 15_gather_token ===\n\n");
    show_seq(tokens, T_IN, "INPUT sequence (6 tokens, dim=4)");

    printf("\nGather indices: [%d, %d, %d]\n",
           indices[0], indices[1], indices[2]);

    gather_tokens(tokens, indices, out, T_OUT);

    printf("\n");
    show_seq(out, T_OUT, "OUTPUT (3 selected tokens)");

    /* Verify: out[1] should be a copy of tokens[2] */
    int ok = 1;
    for (d = 0; d < D; d++)
    {
        if (out[1 * D + d] != tokens[2 * D + d])
        {
            ok = 0;
        }
    }
    printf("\nCHECK: out[1] == tokens[indices[1]=2] → %s\n", ok ? "MATCH" : "MISMATCH");

    /* VLA analogy: last 3 tokens of a 6-token sequence = action tokens */
    int action_indices[3] = {3, 4, 5};
    float action_out[3 * D];
    gather_tokens(tokens, action_indices, action_out, 3);
    printf("\nVLA analogy — gather last 3 tokens (action tokens):\n");
    show_seq(action_out, 3, "  action tokens");

    return 0;
}
