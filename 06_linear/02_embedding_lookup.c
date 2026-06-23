/*
 * 02_embedding_lookup.c
 *
 * Op: Token ID → dense vector lookup.
 *     indices[N][T] (integers)  →  output[N][T][D]
 *     out[n][t] = table[ indices[n][t] ]   (row copy, no multiply)
 *
 * Table  layout: [V][D]    (vocab_size × embed_dim), flat: v*D + d
 * Output layout: [N][T][D], flat: n*T*D + t*D + d
 *
 * Comment: "Embedding lookup = table indexing, not matrix multiply.
 *  Index is a token ID; the lookup row IS the token's vector.
 *  Dynamic T (sequence length) is the key runtime variable in all
 *  LLM-based VLA models."
 *
 * WHERE IN MODELS:
 *   - ALL transformer-based models: BERT, GPT, LLaMA word embeddings.
 *   - ViT patch embed: the linear projection that follows patch extraction
 *     is a learned weight matrix — same op as this once tokens are formed.
 *   - VLA language encoders (RT-2, OpenVLA): first op on the token sequence;
 *     the NLP side embeds word IDs before cross-attending to vision tokens.
 *   - Sequence length T is DYNAMIC: changes per input (prompt length, number
 *     of detected objects, etc.) — the canonical runtime-shape scenario.
 *
 * BUILD:
 *   gcc -O2 -o 02_embedding_lookup 02_embedding_lookup.c -lm && ./02_embedding_lookup
 */

#include <stdio.h>
#include <math.h>

/* Static shapes for this demo */
#define V   6   /* vocab size  */
#define D   4   /* embed dim   */
#define N   2   /* batch size  */
#define T   3   /* seq length  — this is the KEY DYNAMIC variable at runtime */

/* ------------------------------------------------------------------ */
/* show helpers                                                         */
/* ------------------------------------------------------------------ */

static void show_table(const float *tbl)
{
    int v, d;
    printf("  Embedding table [V=%d][D=%d]:\n", V, D);
    for (v = 0; v < V; v++)
    {
        printf("  row %d: [", v);
        for (d = 0; d < D; d++)
        {
            /* flat: v*D + d */
            printf(" %5.1f", tbl[v * D + d]);
        }
        printf(" ]\n");
    }
}

static void show_indices(const int *idx)
{
    int n, t;
    printf("  Token indices [N=%d][T=%d]:\n", N, T);
    for (n = 0; n < N; n++)
    {
        printf("  n=%d: [", n);
        for (t = 0; t < T; t++)
        {
            /* flat: n*T + t */
            printf(" %d", idx[n * T + t]);
        }
        printf(" ]\n");
    }
}

static void show_output(const float *out)
{
    int n, t, d;
    printf("  Output [N=%d][T=%d][D=%d]:\n", N, T, D);
    for (n = 0; n < N; n++)
    {
        for (t = 0; t < T; t++)
        {
            printf("  n=%d t=%d: [", n, t);
            for (d = 0; d < D; d++)
            {
                /* flat: n*T*D + t*D + d */
                printf(" %5.1f", out[n * T * D + t * D + d]);
            }
            printf(" ]\n");
        }
    }
}

/* ------------------------------------------------------------------ */
/* embedding_lookup                                                     */
/*                                                                      */
/* Dynamic strides (computed at runtime, no compile-time constants):    */
/*   int stride_n_idx = t_dim;          step to next batch in indices  */
/*   int stride_n_out = t_dim * d_dim;  step to next batch in output   */
/*   int stride_t_out = d_dim;          step to next token in output   */
/*   int stride_v     = d_dim;          step to next vocab row in table */
/* ------------------------------------------------------------------ */
static void embedding_lookup(const float *table,  /* [V][D]    */
                             const int   *indices, /* [N][T]    */
                             float       *out,     /* [N][T][D] */
                             int v_dim, int d_dim, int n_dim, int t_dim)
{
    /* dynamic strides: computed at runtime, no compile-time constants */
    int stride_n_idx = t_dim;          /* step to next batch item in indices */
    int stride_n_out = t_dim * d_dim;  /* step to next batch item in output  */
    int stride_t_out = d_dim;          /* step to next token in output       */
    int stride_v     = d_dim;          /* step to next row in embedding table */

    (void)v_dim; /* used for documentation; bounds not checked in this demo */

    int n, t, d;
    for (n = 0; n < n_dim; n++)
    {
        for (t = 0; t < t_dim; t++)
        {
            /* indices[n][t]: n*stride_n_idx + t */
            int tok = indices[n * stride_n_idx + t]; /* token ID (row in table) */
            for (d = 0; d < d_dim; d++)
            {
                /* table[tok][d]: tok*stride_v + d                       */
                /* out[n][t][d]:  n*stride_n_out + t*stride_t_out + d    */
                out[n * stride_n_out + t * stride_t_out + d] =
                    table[tok * stride_v + d];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    int v, d, i;

    /*
     * Embedding table [V=6][D=4]:
     * row v, dim d  →  value = v*10 + d
     * row 0: [ 0  1  2  3 ]
     * row 1: [10 11 12 13 ]
     * ...
     * row 5: [50 51 52 53 ]
     */
    float table[V * D];
    for (v = 0; v < V; v++)
    {
        for (d = 0; d < D; d++)
        {
            /* flat: v*D + d */
            table[v * D + d] = (float)(v * 10 + d);
        }
    }

    /*
     * Token indices [N=2][T=3]:
     *   batch 0: tokens [0, 2, 4]
     *   batch 1: tokens [1, 3, 5]
     * flat: n*T + t
     */
    int indices[N * T] = {
        0, 2, 4,   /* n=0 */
        1, 3, 5    /* n=1 */
    };

    float out[N * T * D];
    for (i = 0; i < N * T * D; i++)
    {
        out[i] = 0.0f;
    }

    printf("=== 02_embedding_lookup ===\n\n");

    printf("EMBEDDING TABLE:\n");
    show_table(table);

    printf("\nTOKEN INDICES:\n");
    show_indices(indices);

    embedding_lookup(table, indices, out, V, D, N, T);

    printf("\nOUTPUT (each row = looked-up embedding vector):\n");
    show_output(out);

    /* Check 1: out[0][0] = table[indices[0][0]] = table[0] = [0,1,2,3] */
    printf("\nCHECK 1: out[0][0] = [");
    for (d = 0; d < D; d++)
    {
        /* flat: 0*T*D + 0*D + d */
        printf(" %.0f", out[0 * T * D + 0 * D + d]);
    }
    printf(" ]  (expect [ 0 1 2 3 ] = table[0])\n");

    /* Check 2: out[1][2] = table[indices[1][2]] = table[5] = [50,51,52,53] */
    printf("CHECK 2: out[1][2] = [");
    for (d = 0; d < D; d++)
    {
        /* flat: 1*T*D + 2*D + d */
        printf(" %.0f", out[1 * T * D + 2 * D + d]);
    }
    printf(" ]  (expect [ 50 51 52 53 ] = table[5])\n");

    return 0;
}
