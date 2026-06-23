/*
 * Format  : IEEE 754 Single Precision (FP32)
 * Layout  : [1 sign][8 exponent][23 mantissa] = 32 bits
 * Bias    : 127  (stored_exp = true_exp + 127)
 * Range   : ±3.4e38
 * Precision: ~7 decimal digits (2^-23 ≈ 1.19e-7 unit in last place)
 * Special : exp=0,mant=0 → zero | exp=0,mant≠0 → subnormal
 *           exp=255,mant=0 → ±inf | exp=255,mant≠0 → NaN
 * GPU     : NVIDIA A100 trains in FP32 (outer loop) mixed with TF32 (Tensor Core)
 *
 * DEMO    : Bit anatomy, special values, machine epsilon
 * Build   : gcc 01_fp32_basics.c -o 01_fp32_basics -lm
 *
 * FP32 is the training baseline. All lower-precision formats trade range or
 * precision for 2-4x memory and compute savings. NVIDIA A100 trains in FP32
 * mixed with TF32.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * print_fp32_bits
 * Extract the three IEEE 754 fields from the 32-bit representation and
 * display them with the decoded biased exponent value.
 * ----------------------------------------------------------------------- */
void print_fp32_bits(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);           /* safe type-pun: copy bytes, no UB  */

    /* isolate the sign bit: shift right 31 positions */
    uint32_t sign = (bits >> 31) & 0x1;

    /* isolate the exponent: shift right 23 to move it into low bits,
       then mask 8 bits                                                    */
    uint32_t exp_raw = (bits >> 23) & 0xFF;

    /* isolate the mantissa: mask the bottom 23 bits                      */
    uint32_t mant = bits & 0x7FFFFF;

    /* decode the biased exponent (subtract the FP32 bias of 127)         */
    int exp_biased = (int)exp_raw - 127;

    /* print the three fields side-by-side in binary */
    printf("sign=%u exp=", sign);
    for (int i = 7; i >= 0; i--)
    {
        printf("%u", (exp_raw >> i) & 1);
    }
    printf("(bias=%+d) mant=", exp_biased);
    for (int i = 22; i >= 0; i--)
    {
        printf("%u", (mant >> i) & 1);
    }
    printf(" → value=%g\n", f);
}

/* -----------------------------------------------------------------------
 * show_fp32_special_values
 * Print the bit patterns for the IEEE 754 special cases so the reader can
 * see exactly which bit patterns encode them.
 * ----------------------------------------------------------------------- */
void show_fp32_special_values(void)
{
    printf("\n--- FP32 special values ---\n");

    float vals[] = { 0.0f, -0.0f, 1.0f/0.0f, -1.0f/0.0f,
                     0.0f/0.0f, 1.0f, -1.0f, 0.5f, 1.5f };
    const char *labels[] = { "+0", "-0", "+inf", "-inf",
                              "NaN", "1.0", "-1.0", "0.5", "1.5" };
    int n = 9;

    for (int i = 0; i < n; i++)
    {
        printf("  %-6s  ", labels[i]);
        print_fp32_bits(vals[i]);
    }
}

int main(void)
{
    printf("=== 01_fp32_basics: IEEE 754 Single Precision ===\n");
    printf("FP32: 1 sign + 8 exp + 23 mantissa = 32 bits | range +-3.4e38 | ~7 decimal digits\n\n");

    /* --- Bit anatomy for a set of representative values --- */
    printf("--- Bit anatomy ---\n");
    float demo[] = { 0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 3.14f, 1e38f,
                     1.0f/0.0f, 0.0f/0.0f };
    const char *dlabels[] = { "0.0", "1.0", "-1.0", "0.5", "1.5",
                               "3.14", "1e38", "inf", "NaN" };
    int nd = 9;

    for (int i = 0; i < nd; i++)
    {
        printf("  %-6s  ", dlabels[i]);
        print_fp32_bits(demo[i]);
    }

    /* --- Subnormal: smallest positive FP32 value --- */
    printf("\n--- Subnormal (exp=0, mant=1) ---\n");
    printf("  1.4e-45  ");
    print_fp32_bits(1.4e-45f);   /* 0 00000000 00000000000000000000001 */

    /* --- Machine epsilon: smallest e such that 1.0f + e != 1.0f --- */
    printf("\n--- Machine epsilon ---\n");
    float eps = 1.0f;
    while (1.0f + (eps / 2.0f) != 1.0f)
    {
        eps = eps / 2.0f;       /* halve until we can no longer distinguish */
    }
    printf("  machine epsilon = %.6e  (≈ 2^-23 = 1.19e-7)\n", eps);
    printf("  bits of epsilon: ");
    print_fp32_bits(eps);

    /* --- Special values tour --- */
    show_fp32_special_values();

    /* --- Check: 1.0f must be 0_01111111_00000000000000000000000 --- */
    printf("\n--- CHECK ---\n");
    uint32_t one_bits;
    memcpy(&one_bits, &(float){1.0f}, 4);
    /* expected: sign=0, exp=127 (01111111), mant=0 */
    uint32_t expected = (0u << 31) | (127u << 23) | 0u;
    if (one_bits == expected)
    {
        printf("  PASS: 1.0f bits = 0_01111111_00000000000000000000000\n");
    }
    else
    {
        printf("  FAIL: 1.0f bits = %08X (expected %08X)\n", one_bits, expected);
    }

    return 0;
}
