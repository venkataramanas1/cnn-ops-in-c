/* ============================================================================
 * [39] FPN Output 3x3  --  the smoothing conv after a top-down merge
 * ----------------------------------------------------------------------------
 * SHAPE: a 3x3 square applied AFTER two feature maps have been added together
 * in the FPN top-down pathway. Its job is to smooth and integrate the merged
 * information. When you add a upsampled deep feature to a shallow feature, the
 * result can have a "checkerboard" or "staircase" artifact. The 3x3 blurs it.
 *
 *   Input : [C_fpn][H][W]            (the merged add from [38])
 *   Weight: [C_fpn][C_fpn][3][3]     (same channel count in and out)
 *   Output: [C_fpn][H][W]            (smoothed output feature level)
 *
 * FLOPs: 2 * C_fpn * C_fpn * H * W * 9
 *
 * DEMO: a merged map with a checkerboard pattern (alternating high/low values,
 * the artifact from nearest-neighbor upsample). The 3x3 averaging kernel
 * smooths it out -- compare before and after.
 *
 * Build:  gcc 39_fpn_output_3x3.c -o 39_fpn_output_3x3 -lm
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>

#define K   3
#define PAD 1

static void show(const char *title, const float *m, int C, int H, int W)
{
    printf("%s  [C=%d H=%d W=%d]\n", title, C, H, W);
    for (int c = 0; c < C; c++) {
        if (C > 1) { printf("  channel %d:\n", c); }
        for (int h = 0; h < H; h++) {
            printf("   ");
            for (int w = 0; w < W; w++) { printf("%7.3f", m[c*H*W+h*W+w]); }
            printf("\n");
        }
    }
    printf("\n");
}

static void conv3x3(const float *in, const float *w, float *out,
                    int Ci, int Co, int H, int W)
{
    /* FPN output 3x3: smoothing conv applied after the top-down element-wise add.
       Nearest-neighbor upsample produces checkerboard / staircase artifacts;
       this 3x3 integrates the neighbourhood and blurs them away.
       Same-padding (PAD=1) keeps the spatial size identical. */
    for (int co=0;co<Co;co++) {         /* co: output channel */
        for (int h=0;h<H;h++) {         /* h: output row */
            for (int x=0;x<W;x++) {     /* x: output column */
                float s=0.0f;           /* accumulator for this output pixel */
                for (int ci=0;ci<Ci;ci++) {   /* ci: input channel */
                    for (int kh=0;kh<K;kh++) {    /* kh: kernel row (0..K-1) */
                        for (int kw=0;kw<K;kw++) { /* kw: kernel col (0..K-1) */
                            int ih=h+kh-PAD, iw=x+kw-PAD; /* input coords with pad offset */
                            if (ih<0||ih>=H||iw<0||iw>=W) { continue; } /* outside image boundary = zero padding */
                            /* input flat index: [ci][ih][iw] -> ci*H*W + ih*W + iw */
                            /* weight flat index: [co][ci][kh][kw] -> ((co*Ci+ci)*K+kh)*K+kw */
                            s += in[ci*H*W+ih*W+iw]*w[((co*Ci+ci)*K+kh)*K+kw];
                        }
                    }
                }
                out[co*H*W+h*W+x]=s; /* smoothed output pixel at [co][h][x] */
            }
        }
    }
}

int main(void)
{
    int C = 1, H = 6, W = 6;

    float *in  = calloc(C*H*W,   sizeof(float));
    float *w   = calloc(C*C*K*K, sizeof(float));
    float *out = calloc(C*H*W,   sizeof(float));

    /* checkerboard artifact: alternating 8.0 and 1.0 */
    for (int h=0;h<H;h++) {
        for (int x=0;x<W;x++) {
            in[h*W+x] = ((h+x) % 2 == 0) ? 8.0f : 1.0f;
        }
    }
    /* averaging box kernel 1/9 */
    for (int i=0;i<C*C*K*K;i++) { w[i] = 1.0f/9.0f; }

    show("INPUT  (checkerboard artifact from upsample)", in, C, H, W);
    show("KERNEL (3x3 averaging = the smoothing conv)", w, 1, K, K);
    conv3x3(in, w, out, C, C, H, W);
    show("OUTPUT (smoothed -- artifact reduced)", out, C, H, W);

    printf("[39] check: interior cell should be ~4.5 (avg of 8/1 pattern)\n");
    printf("[39] out center = %.3f\n", out[2*W+2]);

    free(in); free(w); free(out);
    return 0;
}
