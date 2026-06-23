/*
 * Format  : IEEE 754 Half Precision (FP16)
 * Layout  : [1 sign][5 exponent][10 mantissa] = 16 bits
 * Bias    : 15  (stored_exp = true_exp + 15)
 * Range   : ±65504  |  min normal ≈ 6.1e-5
 * Precision: ~3 decimal digits
 * Overflow: values > 65504 become inf (gradient overflow risk!)
 * GPU     : NVIDIA V100/A100 Tensor Cores for mixed-precision training
 *
 * DEMO    : fp32↔fp16 round-trip, overflow demonstration, precision loss
 * Build   : gcc 02_fp16_basics.c -o 02_fp16_basics -lm
 *
 * FP16 is used in NVIDIA V100/A100 Tensor Cores for mixed-precision training.
 * Risk: overflow at >65504. Gradients can overflow → NaN. BF16 solves this.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * fp32_to_fp16
 * Manual conversion: re-bias the exponent (FP32 bias=127 → FP16 bias=15,
 * so subtract 112), then truncate the 23-bit mantissa to 10 bits.
 * Returns the 16-bit FP16 representation as uint16_t.
 * ----------------------------------------------------------------------- */
uint16_t fp32_to_fp16(float f)
{
    uint32_t src;
    memcpy(&src, &f, 4);           /* safe byte-copy: read FP32 bits     */

    /* extract FP32 fields */
    uint32_t sign32  = (src >> 31) & 0x1;          /* 1 bit              */
    uint32_t exp32   = (src >> 23) & 0xFF;          /* 8 bits             */
    uint32_t mant32  = src & 0x7FFFFF;              /* 23 bits            */

    uint16_t h;

    /* handle FP32 special cases before re-biasing */
    if (exp32 == 0xFF)
    {
        /* inf or NaN: exp16 = 0x1F (all 1s), preserve NaN mantissa bit   */
        uint16_t mant16 = (mant32 != 0) ? 0x200 : 0x000;  /* NaN or inf  */
        h = (uint16_t)((sign32 << 15) | (0x1F << 10) | mant16);
        return h;
    }

    if (exp32 == 0 && mant32 == 0)
    {
        /* ±zero: preserve sign, zero everything else                     */
        h = (uint16_t)(sign32 << 15);
        return h;
    }

    /* re-bias: FP32 bias=127, FP16 bias=15, difference=112              */
    int exp16_val = (int)exp32 - 127 + 15;

    if (exp16_val >= 0x1F)
    {
        /* exponent too large: overflow → ±inf                           */
        h = (uint16_t)((sign32 << 15) | (0x1F << 10));
        return h;
    }

    if (exp16_val <= 0)
    {
        /* underflow: would become FP16 subnormal or zero; clamp to zero  */
        h = (uint16_t)(sign32 << 15);
        return h;
    }

    /* truncate mantissa from 23 bits down to 10 bits (round-to-nearest) */
    uint16_t mant16 = (uint16_t)(mant32 >> 13);    /* drop bottom 13 bits */

    /* pack: [sign(1)][exp(5)][mant(10)]                                  */
    h = (uint16_t)((sign32 << 15) | ((uint16_t)exp16_val << 10) | mant16);
    return h;
}

/* -----------------------------------------------------------------------
 * fp16_to_fp32
 * Reverse: restore FP32 bias from FP16 bias (add 112), zero-extend
 * the 10-bit mantissa back to 23 bits.
 * ----------------------------------------------------------------------- */
float fp16_to_fp32(uint16_t h)
{
    uint32_t sign16  = (h >> 15) & 0x1;
    uint32_t exp16   = (h >> 10) & 0x1F;           /* 5-bit exponent      */
    uint32_t mant16  = h & 0x3FF;                  /* 10-bit mantissa     */

    uint32_t dst;

    if (exp16 == 0x1F)
    {
        /* FP16 inf or NaN: reproduce in FP32 (exp=0xFF)                  */
        uint32_t mant32 = (mant16 != 0) ? 0x400000 : 0;
        dst = (sign16 << 31) | (0xFF << 23) | mant32;
    }
    else if (exp16 == 0 && mant16 == 0)
    {
        /* ±zero */
        dst = sign16 << 31;
    }
    else
    {
        /* normal: re-bias exponent from FP16 (bias=15) back to FP32 (bias=127) */
        uint32_t exp32  = (uint32_t)((int)exp16 - 15 + 127);
        uint32_t mant32 = mant16 << 13;             /* zero-extend 10→23   */
        dst = (sign16 << 31) | (exp32 << 23) | mant32;
    }

    float result;
    memcpy(&result, &dst, 4);      /* safe byte-copy back to float       */
    return result;
}

/* -----------------------------------------------------------------------
 * print_fp16_bits
 * Display sign, 5-bit exponent with decoded value, and 10-bit mantissa.
 * ----------------------------------------------------------------------- */
void print_fp16_bits(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    int exp_val   = (int)exp - 15;           /* biased value (FP16 bias=15) */

    printf("sign=%u exp=", sign);
    for (int i = 4; i >= 0; i--)
    {
        printf("%u", (exp >> i) & 1);
    }
    printf("(val=%+d) mant=", exp_val);
    for (int i = 9; i >= 0; i--)
    {
        printf("%u", (mant >> i) & 1);
    }
    printf(" → FP32=%g", fp16_to_fp32(h));
}

int main(void)
{
    printf("=== 02_fp16_basics: IEEE 754 Half Precision (FP16) ===\n");
    printf("FP16: 1 sign + 5 exp + 10 mantissa = 16 bits | range +-65504 | ~3 decimal digits\n\n");

    /* round-trip demo array */
    float inputs[] = { 1.0f, 0.5f, 0.1f, 100.0f, 65504.0f,
                       65505.0f, 1e-5f };
    const char *labels[] = { "1.0", "0.5", "0.1", "100.0",
                              "65504.0", "65505.0(overflow?)", "1e-5(sub?)" };
    int n = 7;

    printf("%-22s  %-18s  %-17s  %s\n",
           "input", "fp16 bits", "dequantized", "error");
    printf("%-22s  %-18s  %-17s  %s\n",
           "------", "---------", "-----------", "-----");

    for (int i = 0; i < n; i++)
    {
        uint16_t h   = fp32_to_fp16(inputs[i]);
        float    deq = fp16_to_fp32(h);
        float    err = fabsf(inputs[i] - deq);

        printf("  %-20s  ", labels[i]);
        print_fp16_bits(h);
        printf("  |error|=%.4e\n", err);
    }

    /* special demonstration: overflow */
    printf("\n--- Overflow demonstration ---\n");
    printf("  fp16 max = 65504\n");
    printf("  65504.0 → fp16 → ");
    print_fp16_bits(fp32_to_fp16(65504.0f));
    printf("\n");
    printf("  65505.0 → fp16 → ");
    print_fp16_bits(fp32_to_fp16(65505.0f));
    printf("  (becomes inf!)\n");

    /* precision comparison for 0.1 */
    printf("\n--- Precision comparison ---\n");
    float   orig  = 0.1f;
    float   deq01 = fp16_to_fp32(fp32_to_fp16(orig));
    printf("  0.1 in FP32: %.6f, in FP16: ~%.6f, error: %.2e\n",
           orig, deq01, fabsf(orig - deq01));

    /* CHECK: 1.0f round-trip must be exact */
    printf("\n--- CHECK ---\n");
    float rt = fp16_to_fp32(fp32_to_fp16(1.0f));
    if (rt == 1.0f)
    {
        printf("  PASS: fp16_to_fp32(fp32_to_fp16(1.0f)) == 1.0f\n");
    }
    else
    {
        printf("  FAIL: round-trip of 1.0f gave %g\n", rt);
    }

    return 0;
}
