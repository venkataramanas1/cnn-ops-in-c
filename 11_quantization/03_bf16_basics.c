/*
 * Format  : Brain Float 16 (BF16) — Google Brain / NVIDIA
 * Layout  : [1 sign][8 exponent][7 mantissa] = 16 bits
 * Bias    : 127  (identical to FP32 — the KEY design choice)
 * Range   : ±3.4e38  (same as FP32 — no overflow risk unlike FP16!)
 * Precision: ~2 decimal digits  (FP16 has ~3 — BF16 trades precision for range)
 * Concept : BF16 is literally the top 16 bytes of a FP32 with rounding
 * GPU     : NVIDIA A100 Tensor Cores (native BF16). Used in LLaMA, RT-2, OpenVLA.
 *
 * DEMO    : fp32↔bf16 round-trip, overflow immunity, precision comparison vs FP16
 * Build   : gcc 03_bf16_basics.c -o 03_bf16_basics -lm
 *
 * BF16 = FP32 with 16 bits of mantissa thrown away. No overflow risk because
 * exp range = FP32. NVIDIA A100 Tensor Cores support BF16. Used in LLaMA,
 * RT-2, OpenVLA training. Tradeoff: less precision than FP16 near 1.0,
 * but never overflows.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * fp32_to_bf16
 * BF16 shares the same exponent field as FP32 (bias=127, 8 bits), so
 * conversion is simply: keep the top 16 bits, round the result.
 * Round-to-nearest-even: inspect bit 16 and the sticky bits below.
 * ----------------------------------------------------------------------- */
uint16_t fp32_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);           /* read raw FP32 bits safely          */

    /* check for NaN: preserve NaN-ness (avoid turning NaN into inf)       */
    if (((bits >> 23) & 0xFF) == 0xFF && (bits & 0x7FFFFF) != 0)
    {
        /* canonical NaN in BF16: exp=0xFF, mant bit 6 = 1                */
        return (uint16_t)((bits >> 16) | 0x0040);
    }

    /* round-to-nearest: the rounding bit is bit 15 of the FP32 word      */
    uint32_t rounding_bit = (bits >> 15) & 0x1;
    /* sticky bits: any of bits 14..0 set?                                 */
    uint32_t sticky = bits & 0x7FFF;

    /* add the rounding increment to the upper 16 bits                    */
    uint32_t bf16_bits = bits >> 16;
    if (rounding_bit && sticky)
    {
        bf16_bits += 1;             /* round up (ties go to nearest even)  */
    }
    else if (rounding_bit && !sticky)
    {
        /* exactly halfway: round to even — increment if mantissa is odd  */
        bf16_bits += (bf16_bits & 1);
    }

    return (uint16_t)bf16_bits;
}

/* -----------------------------------------------------------------------
 * bf16_to_fp32
 * Trivially reverse: zero-extend the 16 bits back into the top half of a
 * 32-bit word (the lower 16 bits are just zeros — the truncated mantissa).
 * ----------------------------------------------------------------------- */
float bf16_to_fp32(uint16_t b)
{
    uint32_t bits = ((uint32_t)b) << 16;    /* place BF16 in top 16 bits  */
    float result;
    memcpy(&result, &bits, 4);              /* safe byte-copy back         */
    return result;
}

/* -----------------------------------------------------------------------
 * print_bf16_bits
 * Show sign, 8-bit exponent (same bias=127 as FP32), and 7-bit mantissa.
 * ----------------------------------------------------------------------- */
void print_bf16_bits(uint16_t b)
{
    uint32_t sign = (b >> 15) & 0x1;
    uint32_t exp  = (b >>  7) & 0xFF;       /* 8-bit exponent              */
    uint32_t mant = b & 0x7F;               /* 7-bit mantissa              */
    int exp_val   = (int)exp - 127;          /* un-bias (same as FP32)     */

    printf("sign=%u exp=", sign);
    for (int i = 7; i >= 0; i--)
    {
        printf("%u", (exp >> i) & 1);
    }
    printf("(bias=%+d) mant=", exp_val);
    for (int i = 6; i >= 0; i--)
    {
        printf("%u", (mant >> i) & 1);
    }
    printf(" → FP32=%g", bf16_to_fp32(b));
}

/* forward-declare the FP16 functions for side-by-side comparison */
uint16_t fp32_to_fp16_ref(float f);
float    fp16_to_fp32_ref(uint16_t h);

int main(void)
{
    printf("=== 03_bf16_basics: Brain Float 16 (BF16) ===\n");
    printf("BF16: 1 sign + 8 exp + 7 mantissa = 16 bits | range +-3.4e38 | ~2 decimal digits\n\n");

    /* round-trip demo — same array as FP16 for fair comparison */
    float inputs[] = { 1.0f, 0.5f, 0.1f, 100.0f, 65504.0f,
                       65505.0f, 1e-5f };
    const char *labels[] = { "1.0", "0.5", "0.1", "100.0",
                              "65504.0", "65505.0", "1e-5" };
    int n = 7;

    printf("--- BF16 round-trip ---\n");
    for (int i = 0; i < n; i++)
    {
        uint16_t b   = fp32_to_bf16(inputs[i]);
        float    deq = bf16_to_fp32(b);
        float    err = fabsf(inputs[i] - deq);

        printf("  %-10s  ", labels[i]);
        print_bf16_bits(b);
        printf("  |error|=%.4e\n", err);
    }

    /* KEY demo: 1e38 overflow in FP16 vs safe in BF16 */
    printf("\n--- KEY: overflow immunity ---\n");
    {
        float big = 1e38f;

        /* BF16: should stay finite because exp range = FP32             */
        uint16_t bf = fp32_to_bf16(big);
        float    bf_deq = bf16_to_fp32(bf);
        printf("  1e38 → BF16 → %g  (finite? %s)\n",
               bf_deq, isfinite(bf_deq) ? "YES" : "NO");

        /* reference: simulate FP16 overflow manually                    */
        printf("  1e38 → FP16 → inf  (FP16 max=65504, this overflows)\n");
    }

    /* precision comparison near 1.0: FP16 has 10 mantissa bits,
       BF16 has only 7, so BF16 is coarser near 1.0                      */
    printf("\n--- Side-by-side precision table ---\n");
    printf("  %-12s  %-14s  %-14s  %-14s  %-14s\n",
           "value", "fp16_result", "bf16_result", "fp16_error", "bf16_error");
    printf("  %-12s  %-14s  %-14s  %-14s  %-14s\n",
           "-----", "-----------", "-----------", "----------", "----------");

    float cmp[] = { 1.0f, 0.1f, 3.14159f, 0.001f, 1000.0f };
    int nc = 5;

    for (int i = 0; i < nc; i++)
    {
        /* inline minimal fp16 for comparison (no extra file needed) */
        float f = cmp[i];

        /* fp16 conversion inline */
        uint32_t src;
        memcpy(&src, &f, 4);
        uint32_t s16 = (src >> 31) & 0x1;
        uint32_t e32 = (src >> 23) & 0xFF;
        uint32_t m32 = src & 0x7FFFFF;
        float fp16_deq;
        if (e32 == 0xFF)
        {
            uint32_t d = (s16 << 31) | (0xFF << 23) | (m32 ? 0x400000 : 0);
            memcpy(&fp16_deq, &d, 4);
        }
        else
        {
            int e16v = (int)e32 - 127 + 15;
            if (e16v >= 0x1F)
            {
                uint32_t d = (s16 << 31) | (0xFF << 23);
                memcpy(&fp16_deq, &d, 4);
            }
            else if (e16v <= 0)
            {
                uint32_t d = s16 << 31;
                memcpy(&fp16_deq, &d, 4);
            }
            else
            {
                uint16_t m16 = (uint16_t)(m32 >> 13);
                uint16_t h   = (uint16_t)((s16 << 15) | ((uint16_t)e16v << 10) | m16);
                uint32_t exp_out = (uint32_t)((int)((h >> 10) & 0x1F) - 15 + 127);
                uint32_t d = (s16 << 31) | (exp_out << 23) | ((uint32_t)(h & 0x3FF) << 13);
                memcpy(&fp16_deq, &d, 4);
            }
        }

        float bf16_deq = bf16_to_fp32(fp32_to_bf16(f));
        float fp16_err = fabsf(f - fp16_deq);
        float bf16_err = fabsf(f - bf16_deq);

        printf("  %-12g  %-14g  %-14g  %-14.4e  %-14.4e\n",
               f, fp16_deq, bf16_deq, fp16_err, bf16_err);
    }

    printf("\n  3.14159 in FP32 has 7 sig digits; BF16 cuts mantissa to 7 bits → only ~2 sig digits\n");

    /* CHECK: bf16 of 1e38 must NOT be inf */
    printf("\n--- CHECK ---\n");
    float bf_1e38 = bf16_to_fp32(fp32_to_bf16(1e38f));
    if (isfinite(bf_1e38))
    {
        printf("  PASS: fp32_to_bf16(1e38f) does NOT produce inf (got %g)\n", bf_1e38);
    }
    else
    {
        printf("  FAIL: fp32_to_bf16(1e38f) incorrectly produced inf\n");
    }

    return 0;
}
