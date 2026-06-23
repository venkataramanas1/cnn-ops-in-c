/*
 * 08_int8_asymmetric.c — Asymmetric INT8 (uint8) quantization
 *
 * Scheme: scale + zero_point to cover the full [min, max] float range.
 *   scale      = (max(x) - min(x)) / 255.0f
 *   zero_point = round(-min(x) / scale)     — integer so 0.0f -> zero_point
 *   q[i]       = clamp(round(x[i] / scale) + zero_point, 0, 255)   uint8
 *   x_deq[i]   = (q[i] - zero_point) * scale
 *
 * Asymmetric: maps [min,max] to [0,255] using zero_point. Activations after
 * ReLU have min=0, so ZP=0 — degenerates to symmetric. But for layers with
 * asymmetric ranges (e.g. tanh output [-1,1]), asymmetric gives better coverage.
 *
 * Models: TFLite default quantization (activations are asymmetric uint8),
 *         ONNX Runtime quantize_linear op
 * Build:  gcc 08_int8_asymmetric.c -o 08_int8_asymmetric -lm
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ------------------------------------------------------------------ helpers */

static float clampf(float v, float lo, float hi)
{
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

/* ------------------------------------------------------------------ API */

/*
 * compute_scale_zp — derive scale and zero_point from tensor min/max
 *   scale      = (max - min) / 255.0f
 *   zero_point = round(-min / scale)   clamped to [0, 255]
 */
void compute_scale_zp(const float *x, int n, float *scale, int *zp)
{
    float xmin =  1e30f;
    float xmax = -1e30f;
    int i;
    for (i = 0; i < n; i++)
    {
        if (x[i] < xmin) { xmin = x[i]; }
        if (x[i] > xmax) { xmax = x[i]; }
    }
    *scale = (xmax - xmin) / 255.0f;                   /* map full range to 255 levels */
    float zp_f = roundf(-xmin / (*scale));              /* zero_point: float 0 -> this bucket */
    *zp = (int)clampf(zp_f, 0.0f, 255.0f);
}

/*
 * quantize_asymmetric — float → uint8
 *   q[i] = clamp(round(x[i] / scale) + zero_point, 0, 255)
 */
void quantize_asymmetric(const float *x, uint8_t *q, float scale, int zp, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        float scaled  = x[i] / scale;                  /* normalise to quant grid */
        float shifted = roundf(scaled) + (float)zp;    /* shift so 0.0f lands at zp */
        float clamped = clampf(shifted, 0.0f, 255.0f); /* clip to uint8 range */
        q[i] = (uint8_t)clamped;
    }
}

/*
 * dequantize_asymmetric — uint8 → float
 *   x_deq[i] = (q[i] - zero_point) * scale
 */
void dequantize_asymmetric(const uint8_t *q, float *x_deq, float scale, int zp, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        x_deq[i] = ((float)q[i] - (float)zp) * scale; /* un-shift then un-scale */
    }
}

/* ------------------------------------------------------------------ helper: print a case */

static void run_case(const char *label, const float *x, int n)
{
    float   scale;
    int     zp;
    uint8_t q[16];
    float   x_deq[16];

    compute_scale_zp(x, n, &scale, &zp);
    quantize_asymmetric(x, q, scale, zp, n);
    dequantize_asymmetric(q, x_deq, scale, zp, n);

    printf("  -- %s --\n", label);
    printf("  scale = %.6f   zero_point = %d\n\n", scale, zp);

    printf("  %-6s  %-8s  %-6s  %-10s  %-10s\n",
           "idx", "float", "uint8", "deq_float", "error");
    printf("  %-6s  %-8s  %-6s  %-10s  %-10s\n",
           "---", "-----", "-----", "---------", "-----");

    int i;
    for (i = 0; i < n; i++)
    {
        float err = x[i] - x_deq[i];
        printf("  %-6d  %-8.4f  %-6u  %-10.6f  %-10.6f\n",
               i, x[i], (unsigned)q[i], x_deq[i], err);
    }

    /* Check: 0.0f -> zero_point (only meaningful if 0.0f is in range) */
    float zero_val = 0.0f;
    uint8_t zero_q = 0;
    quantize_asymmetric(&zero_val, &zero_q, scale, zp, 1);
    printf("\n  CHECK 0.0f -> uint8=%u  (expected zero_point=%d): %s\n\n",
           (unsigned)zero_q, zp,
           ((int)zero_q == zp) ? "PASS" : "FAIL");
}

/* ------------------------------------------------------------------ demo */

int main(void)
{
    printf("=== 08_int8_asymmetric: Asymmetric INT8 (uint8) Quantization ===\n\n");

    /* Case 1: ReLU output — min=0 so zero_point should be 0 (degenerates to symmetric) */
    float relu_out[8] = { 0.0f, 0.2f, 0.5f, 0.8f, 1.0f, 1.5f, 2.3f, 3.1f };
    run_case("ReLU output [0, 3.1]  (expect ZP = 0)", relu_out, 8);

    /* Case 2: tanh output — bipolar but asymmetric, ZP will be nonzero */
    float tanh_out[7] = { -0.9f, -0.5f, -0.1f, 0.0f, 0.3f, 0.7f, 0.9f };
    run_case("tanh output [-0.9, 0.9]  (expect ZP != 0)", tanh_out, 7);

    return 0;
}
