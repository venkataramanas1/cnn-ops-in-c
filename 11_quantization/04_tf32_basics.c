/*
 * Format  : TensorFloat-32 (TF32) — NVIDIA A100 exclusive
 * Layout  : [1 sign][8 exponent][10 mantissa] = 19 bits of precision
 *           stored in a 32-bit register (lower 13 bits are unused zeros)
 * Bias    : 127  (identical to FP32)
 * Range   : ±3.4e38  (same as FP32 — full exponent range)
 * Precision: ~3 decimal digits  (same mantissa width as FP16's 10 bits)
 *
 * IMPORTANT: TF32 is NOT a storage format. The programmer writes FP32 code.
 * A100 Tensor Cores automatically round each FP32 input to TF32 internally
 * before the multiply-accumulate. The result is accumulated in FP32.
 * Speedup: ~10x over FP32 matmul. Default for torch.matmul on A100.
 *
 * GPU     : NVIDIA A100 (Ampere architecture) — not available on V100 or H100
 *
 * DEMO    : Show TF32 as FP32 with bottom 13 mantissa bits zeroed,
 *           round-trip error, typical neural-network weight error
 * Build   : gcc 04_tf32_basics.c -o 04_tf32_basics -lm
 *
 * TF32 is invisible to the programmer — you write FP32 code, A100 silently
 * uses TF32 in the Tensor Core. 10x faster than FP32, 2x slower than FP16,
 * but no code changes needed. Default for torch.matmul on A100.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * fp32_to_tf32
 * Keep sign (1 bit) + exponent (8 bits) + TOP 10 mantissa bits.
 * Zero the lower 13 mantissa bits. The result still lives in a uint32_t
 * (same width as FP32) — this mirrors how A100 Tensor Cores treat inputs.
 * ----------------------------------------------------------------------- */
uint32_t fp32_to_tf32(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);           /* read FP32 bit pattern safely        */

    /* mask: keep top 19 bits (sign + exp8 + mant10), zero lower 13 bits  */
    /* lower 13 bits of mantissa: bits [12..0]                            */
    uint32_t tf32_bits = bits & 0xFFFFE000u;    /* 0xFFFFE000 = ~0x1FFF  */

    return tf32_bits;
}

/* -----------------------------------------------------------------------
 * tf32_to_fp32
 * Since TF32 is stored in a 32-bit register with the lower 13 bits already
 * zero, we just interpret those 32 bits as a regular FP32 float.
 * ----------------------------------------------------------------------- */
float tf32_to_fp32(uint32_t t)
{
    float result;
    memcpy(&result, &t, 4);         /* safe byte-copy back to float        */
    return result;
}

/* -----------------------------------------------------------------------
 * print_tf32_bits
 * Show the 32-bit word broken into the TF32 logical fields:
 *   sign(1) | exponent(8) | mantissa_kept(10) | zeroed(13)
 * ----------------------------------------------------------------------- */
void print_tf32_bits(uint32_t t)
{
    uint32_t sign     = (t >> 31) & 0x1;
    uint32_t exp_raw  = (t >> 23) & 0xFF;        /* same position as FP32  */
    uint32_t mant_top = (t >> 13) & 0x3FF;       /* top 10 mantissa bits   */
    uint32_t mant_bot = t & 0x1FFF;              /* should be 0 for TF32   */
    int exp_biased    = (int)exp_raw - 127;

    printf("sign=%u exp=", sign);
    for (int i = 7; i >= 0; i--)
    {
        printf("%u", (exp_raw >> i) & 1);
    }
    printf("(bias=%+d) mant_kept=", exp_biased);
    for (int i = 9; i >= 0; i--)
    {
        printf("%u", (mant_top >> i) & 1);
    }
    printf(" zeroed=%013u", 0);     /* always zero in valid TF32           */
    (void)mant_bot;
    printf(" → FP32=%g", tf32_to_fp32(t));
}

int main(void)
{
    printf("=== 04_tf32_basics: TensorFloat-32 (TF32, NVIDIA A100) ===\n");
    printf("TF32: 1 sign + 8 exp + 10 mantissa = 19 precision bits (stored in 32-bit reg)\n");
    printf("NOT a storage format — A100 Tensor Core rounds FP32 inputs to TF32 automatically\n\n");

    /* --- Show TF32 is FP32 with bottom 13 mantissa bits zeroed --- */
    printf("--- FP32 vs TF32 bit comparison ---\n");
    {
        float demo = 3.14159f;
        uint32_t fp32_bits;
        memcpy(&fp32_bits, &demo, 4);
        uint32_t tf32_bits = fp32_to_tf32(demo);

        printf("  3.14159  FP32: ");
        for (int i = 31; i >= 0; i--)
        {
            if (i == 30 || i == 22)
            {
                printf("|");
            }
            printf("%u", (fp32_bits >> i) & 1);
        }
        printf("\n");

        printf("  3.14159  TF32: ");
        for (int i = 31; i >= 0; i--)
        {
            if (i == 30 || i == 22)
            {
                printf("|");
            }
            if (i < 13)
            {
                printf("0");    /* these 13 bits are zeroed in TF32       */
            }
            else
            {
                printf("%u", (tf32_bits >> i) & 1);
            }
        }
        printf("\n");
        printf("                                               [sig|exp(8)|mant_top(10)|zeroed(13)]\n");
    }

    /* --- Round-trip demo --- */
    printf("\n--- Round-trip: fp32 → tf32 → fp32 ---\n");
    float inputs[] = { 1.0f, 3.14159f, 0.1f, 1e30f,
                       -0.5f, 0.001f, 100.0f };
    const char *labels[] = { "1.0", "3.14159", "0.1", "1e30",
                              "-0.5", "0.001", "100.0" };
    int n = 7;

    printf("  %-10s  %-16s  %-14s  %s\n",
           "input", "tf32_bits(top10)", "dequantized", "abs_error");
    printf("  %-10s  %-16s  %-14s  %s\n",
           "-----", "----------------", "-----------", "---------");

    for (int i = 0; i < n; i++)
    {
        uint32_t t   = fp32_to_tf32(inputs[i]);
        float    deq = tf32_to_fp32(t);
        float    err = fabsf(inputs[i] - deq);

        printf("  %-10s  ", labels[i]);
        print_tf32_bits(t);
        printf("  |err|=%.4e\n", err);
    }

    /* --- Typical neural network weights in [-1, 1]: error vs FP16 --- */
    printf("\n--- Error comparison: TF32 vs FP16 for weights in [-1,1] ---\n");
    printf("  Both have 10 mantissa bits → similar relative error, but TF32 never overflows\n");
    {
        float test_weights[] = { 0.1234f, -0.5678f, 0.9999f, -0.0001f, 0.4321f };
        int nw = 5;
        float total_tf32_err = 0.0f;

        for (int i = 0; i < nw; i++)
        {
            float w   = test_weights[i];
            float tf  = tf32_to_fp32(fp32_to_tf32(w));
            float err = fabsf(w - tf);
            total_tf32_err += err;
            printf("  w=%-9.4f  tf32=%-9.6f  err=%.4e\n", w, tf, err);
        }
        printf("  avg TF32 error over [-1,1] sample: %.4e\n",
               total_tf32_err / nw);
    }

    /* --- A100 context note --- */
    printf("\n--- A100 context ---\n");
    printf("  A100 Tensor Core matmul throughput:\n");
    printf("    FP32 matmul  :  19.5 TFLOPS  (no TF32)\n");
    printf("    TF32 matmul  : ~156 TFLOPS   (10x speedup, automatic)\n");
    printf("    FP16/BF16    : ~312 TFLOPS   (2x over TF32)\n");
    printf("  torch.backends.cudnn.allow_tf32 = True  (default on A100)\n");
    printf("  torch.backends.cuda.matmul.allow_tf32 = True  (default on A100)\n");

    /* --- CHECK: 1.0f is a power of 2, mantissa = 0, round-trip exact --- */
    printf("\n--- CHECK ---\n");
    float rt = tf32_to_fp32(fp32_to_tf32(1.0f));
    if (rt == 1.0f)
    {
        printf("  PASS: tf32_to_fp32(fp32_to_tf32(1.0f)) == 1.0f\n");
    }
    else
    {
        printf("  FAIL: round-trip of 1.0f gave %g\n", rt);
    }

    return 0;
}
