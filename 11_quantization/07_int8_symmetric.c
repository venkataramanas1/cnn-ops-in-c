/*
 * 07_int8_symmetric.c — Symmetric INT8 quantization
 *
 * Scheme: scale around zero, no zero-point offset.
 *   scale = max(|x|) / 127.0f
 *   q[i]  = clamp(round(x[i] / scale), -128, 127)
 *   x_deq = q[i] * scale
 *
 * Symmetric INT8: scale maps [-max, +max] to [-127, 127]. No zero-point offset
 * needed — matmul stays as integer multiply+accumulate. TensorRT uses this for weights.
 *
 * Models: NVIDIA TensorRT INT8 mode, ONNX Runtime int8 EP
 * Build:  gcc 07_int8_symmetric.c -o 07_int8_symmetric -lm
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ helpers */

static float absf(float v) { return v < 0.0f ? -v : v; }

static float clampf(float v, float lo, float hi)
{
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

/* ------------------------------------------------------------------ API */

/*
 * compute_scale_symmetric — one scale for the whole tensor
 *   scale = max(|x[i]|) / 127.0f
 */
float compute_scale_symmetric(const float *x, int n)
{
    float max_abs = 0.0f;
    int i;
    for (i = 0; i < n; i++)
    {
        float a = absf(x[i]);       /* absolute value of each element */
        if (a > max_abs) { max_abs = a; }
    }
    return max_abs / 127.0f;        /* map peak magnitude to 127 */
}

/*
 * quantize_symmetric — float → INT8
 *   q[i] = clamp(round(x[i] / scale), -128, 127)
 */
void quantize_symmetric(const float *x, int8_t *q, float scale, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        float scaled  = x[i] / scale;           /* divide by scale */
        float rounded = roundf(scaled);          /* round to nearest integer */
        float clamped = clampf(rounded, -128.0f, 127.0f);  /* clip to INT8 range */
        q[i] = (int8_t)clamped;
    }
}

/*
 * dequantize_symmetric — INT8 → float
 *   x_deq[i] = q[i] * scale
 */
void dequantize_symmetric(const int8_t *q, float *x_deq, float scale, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        x_deq[i] = (float)q[i] * scale;         /* multiply INT8 value by scale */
    }
}

/*
 * quantization_snr — Signal-to-Noise Ratio in dB
 *   SNR = 20 * log10(rms(orig) / rms(error))
 *   Higher is better; INT8 symmetric typically gives ~40-48 dB.
 */
float quantization_snr(const float *orig, const float *deq, int n)
{
    float sum_sig2  = 0.0f;
    float sum_err2  = 0.0f;
    int i;
    for (i = 0; i < n; i++)
    {
        float err    = orig[i] - deq[i];        /* quantization error per element */
        sum_sig2    += orig[i] * orig[i];        /* accumulate signal power */
        sum_err2    += err * err;                /* accumulate noise power */
    }
    float rms_sig = sqrtf(sum_sig2 / (float)n);
    float rms_err = sqrtf(sum_err2 / (float)n);
    if (rms_err < 1e-12f) { return 999.0f; }    /* no error — perfect */
    return 20.0f * log10f(rms_sig / rms_err);   /* SNR in decibels */
}

/* ------------------------------------------------------------------ demo */

int main(void)
{
    /* Conv weight: 8 floats spanning a modest bipolar range */
    float w[8] = { -1.2f, 0.5f, -0.3f, 0.8f, -0.9f, 0.1f, 0.4f, -0.7f };
    int   n    = 8;

    int8_t q[8];
    float  w_deq[8];

    printf("=== 07_int8_symmetric: Symmetric INT8 Quantization ===\n\n");

    /* Step 1: compute the single per-tensor scale */
    float scale = compute_scale_symmetric(w, n);
    printf("  max(|w|)  = 1.200\n");
    printf("  scale     = max(|w|) / 127 = %.6f\n\n", scale);

    /* Step 2: quantize */
    quantize_symmetric(w, q, scale, n);

    /* Step 3: dequantize */
    dequantize_symmetric(q, w_deq, scale, n);

    /* Step 4: print table */
    printf("  %-6s  %-8s  %-6s  %-10s  %-10s\n",
           "idx", "float", "INT8", "deq_float", "error");
    printf("  %-6s  %-8s  %-6s  %-10s  %-10s\n",
           "---", "-----", "----", "---------", "-----");

    int i;
    for (i = 0; i < n; i++)
    {
        float err = w[i] - w_deq[i];
        printf("  %-6d  %-8.4f  %-6d  %-10.6f  %-10.6f\n",
               i, w[i], (int)q[i], w_deq[i], err);
    }

    /* Step 5: SNR */
    float snr = quantization_snr(w, w_deq, n);
    printf("\n  SNR = 20*log10(rms_signal / rms_error) = %.2f dB\n", snr);

    /* Check: 0.0f must map to INT8 zero in symmetric scheme */
    float zero_in   = 0.0f;
    int8_t zero_q   = 0;
    float  zero_deq = 0.0f;
    quantize_symmetric(&zero_in, &zero_q, scale, 1);
    dequantize_symmetric(&zero_q, &zero_deq, scale, 1);
    printf("\n  CHECK: 0.0f -> q=%d -> deq=%.4f  (symmetric: zero maps to INT8 zero exactly)\n",
           (int)zero_q, zero_deq);
    printf("  %s\n", (zero_q == 0) ? "PASS" : "FAIL");

    return 0;
}
