/* ============================================================================
 * [07] Cosine Embedding Loss  --  Vision-language alignment
 * ----------------------------------------------------------------------------
 * Cosine loss maps embeddings onto a unit sphere; similarity is angle, not
 * magnitude. Critical for vision-language alignment in VLA: the vision tower
 * and language tower are trained so matching (image,text) pairs have cos_sim→1.
 *
 * Formulas:
 *   cos_sim(u, v) = dot(u, v) / (||u|| * ||v||)
 *
 *   loss = 1 - cos_sim              if target = +1  (same class / matching pair)
 *        = max(0, cos_sim - margin) if target = -1  (different class)
 *   margin = 0.0  (minimum margin before dissimilar pairs incur no loss)
 *
 * cos_sim ∈ [-1, 1]:
 *   cos_sim =  1   →  identical direction
 *   cos_sim =  0   →  orthogonal (unrelated)
 *   cos_sim = -1   →  opposite direction
 *
 * Models: cross-modal alignment in VLA, contrastive learning (SimCLR),
 *         face verification, semantic similarity.
 *
 * Layout: pairs of [D]-dimensional embedding vectors
 *
 * Build: gcc 07_cosine_embedding_loss.c -o 07_cosine_embedding_loss -lm
 * ============================================================================ */
#include <stdio.h>
#include <math.h>

#define MARGIN 0.0f

/* ---- L2 norm of a D-dimensional vector ----------------------------------- */
static float l2_norm(const float *v, int D)
{
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += v[d] * v[d];   /* accumulate squared components */
    }
    return sqrtf(sum);        /* sqrt for Euclidean norm */
}

/* ---- dot product of two D-dimensional vectors ---------------------------- */
static float dot(const float *u, const float *v, int D)
{
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += u[d] * v[d];   /* element-wise multiply and sum */
    }
    return sum;
}

/* ---- cosine similarity --------------------------------------------------- */
static float cos_sim(const float *u, const float *v, int D)
{
    float dot_uv = dot(u, v, D);            /* numerator: u · v */
    float norm_u = l2_norm(u, D);           /* denominator: ||u|| */
    float norm_v = l2_norm(v, D);           /* denominator: ||v|| */

    if (norm_u < 1e-9f || norm_v < 1e-9f) {
        return 0.0f;                        /* zero vector: undefined, treat as 0 */
    }
    return dot_uv / (norm_u * norm_v);      /* cos(angle between u and v) */
}

/* ---- cosine embedding loss for a single pair ----------------------------- */
static float cosine_embedding_loss(const float *u, const float *v,
                                   float target, int D)
{
    float cs = cos_sim(u, v, D);

    if (target > 0.0f) {
        /* matching pair: push cos_sim toward 1 */
        return 1.0f - cs;
    } else {
        /* non-matching pair: push cos_sim below margin */
        float val = cs - MARGIN;
        return val > 0.0f ? val : 0.0f;    /* max(0, cos_sim - margin) */
    }
}

/* ---- print a vector ------------------------------------------------------ */
static void show_vec(const char *name, const float *v, int D)
{
    printf("  %s: [", name);
    for (int d = 0; d < D; d++) {
        printf("%5.2f%s", v[d], d < D - 1 ? ", " : "");
    }
    printf("]\n");
}

int main(void)
{
    int D = 4;   /* embedding dimension */

    /* pair 1: identical vectors — should give cos_sim=1, loss=0 */
    float u1[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
    float v1[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
    float target1 = 1.0f;   /* same class (matching pair) */

    /* pair 2: orthogonal vectors — cos_sim=0, loss=1 for matching target */
    float u2[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
    float v2[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
    float target2 = 1.0f;   /* same class; orthogonal is bad → high loss */

    /* pair 3: non-matching pair that is orthogonal — no penalty at margin=0 */
    float u3[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
    float v3[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
    float target3 = -1.0f;  /* different class; cos_sim=0, loss=max(0, 0-0)=0 */

    /* pair 4: non-matching pair that is similar — penalized */
    float u4[4] = { 0.9f, 0.1f, 0.0f, 0.0f };
    float v4[4] = { 0.8f, 0.2f, 0.0f, 0.0f };
    float target4 = -1.0f;  /* different class; but vectors are similar → penalized */

    /* compute */
    float cs1 = cos_sim(u1, v1, D);
    float cs2 = cos_sim(u2, v2, D);
    float cs3 = cos_sim(u3, v3, D);
    float cs4 = cos_sim(u4, v4, D);

    float loss1 = cosine_embedding_loss(u1, v1, target1, D);
    float loss2 = cosine_embedding_loss(u2, v2, target2, D);
    float loss3 = cosine_embedding_loss(u3, v3, target3, D);
    float loss4 = cosine_embedding_loss(u4, v4, target4, D);

    /* display */
    printf("COSINE EMBEDDING LOSS  [D=%d  margin=%.1f]\n\n", D, MARGIN);

    printf("Pair 1: identical vectors (target=+1)\n");
    show_vec("u", u1, D);
    show_vec("v", v1, D);
    printf("  cos_sim=%.4f  loss=%.4f\n\n", cs1, loss1);

    printf("Pair 2: orthogonal vectors (target=+1)\n");
    show_vec("u", u2, D);
    show_vec("v", v2, D);
    printf("  cos_sim=%.4f  loss=%.4f\n\n", cs2, loss2);

    printf("Pair 3: orthogonal vectors (target=-1, different class)\n");
    show_vec("u", u3, D);
    show_vec("v", v3, D);
    printf("  cos_sim=%.4f  loss=%.4f  (cos_sim=0 ≤ margin, so no penalty)\n\n",
           cs3, loss3);

    printf("Pair 4: similar vectors (target=-1, different class) — penalized\n");
    show_vec("u", u4, D);
    show_vec("v", v4, D);
    printf("  cos_sim=%.4f  loss=%.4f\n\n", cs4, loss4);

    /* checks */
    printf("[07] check: identical pair loss=%.4f  (expected 0.0000)\n",  loss1);
    printf("[07] check: orthogonal pair loss=%.4f  (expected 1.0000)\n", loss2);
    printf("[07] check: non-matching ortho loss=%.4f  (expected 0.0000)\n\n",
           loss3);

    return 0;
}
