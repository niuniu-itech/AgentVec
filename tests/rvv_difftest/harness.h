/* Common differential-testing driver.
 * A kernel .c file defines DT, DT_IS_FLOAT (0/1), REDUCE (0/1), optionally
 * OUT_FACTOR for multi-output map kernels, and
 * agentvec_kernel(const DT*,const DT*,const DT*,DT*,int), then #includes this.
 * argv: seed n out_path. Inputs are generated deterministically from seed so
 * the scalar oracle build and the RVV build see identical inputs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n);

#ifndef OUT_FACTOR
#define OUT_FACTOR 1
#endif
#ifndef A_IS_MATRIX
#define A_IS_MATRIX 0
#endif
#ifndef B_IS_MATRIX
#define B_IS_MATRIX 0
#endif
#ifndef C_IS_MATRIX
#define C_IS_MATRIX 0
#endif
#ifndef OUT_IS_MATRIX
#define OUT_IS_MATRIX 0
#endif
#ifndef SYMMETRIC_A
#define SYMMETRIC_A 0
#endif
#ifndef LOWER_TRIANGULAR_A
#define LOWER_TRIANGULAR_A 0
#endif
#ifndef BANDED_A_RADIUS
#define BANDED_A_RADIUS -1
#endif

static uint32_t _st;
static uint32_t _xs(void) { _st ^= _st << 13; _st ^= _st >> 17; _st ^= _st << 5; return _st; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s seed n outpath\n", argv[0]); return 2; }
    uint32_t seed = (uint32_t)strtoul(argv[1], 0, 10);
    int n = atoi(argv[2]);
    const char *outp = argv[3];
    _st = seed ? seed : 1u;

    size_t a_len = A_IS_MATRIX ? (size_t)n * (size_t)n : (size_t)n;
    size_t b_len = B_IS_MATRIX ? (size_t)n * (size_t)n : (size_t)n;
    size_t c_len = C_IS_MATRIX ? (size_t)n * (size_t)n : (size_t)n;
    size_t outlen = REDUCE ? 1u : (OUT_IS_MATRIX ? (size_t)n * (size_t)n : (size_t)n * (size_t)OUT_FACTOR);
    DT *a = malloc(sizeof(DT) * a_len);
    DT *b = malloc(sizeof(DT) * b_len);
    DT *c = malloc(sizeof(DT) * c_len);
    DT *out = calloc((size_t)outlen, sizeof(DT));

    size_t max_len = a_len;
    if (b_len > max_len) max_len = b_len;
    if (c_len > max_len) max_len = c_len;
    for (size_t i = 0; i < max_len; i++) {
#if DT_IS_FLOAT
        if (i < a_len) a[i] = (DT)((int32_t)(_xs() % 20001) - 10000) / (DT)100.0;
        if (i < b_len) b[i] = (DT)((int32_t)(_xs() % 20001) - 10000) / (DT)100.0;
        if (i < c_len) c[i] = (DT)((int32_t)(_xs() % 20001) - 10000) / (DT)100.0;
#else
        if (i < a_len) a[i] = (DT)((int32_t)(_xs() % 2001) - 1000);
        if (i < b_len) b[i] = (DT)((int32_t)(_xs() % 2001) - 1000);
        if (i < c_len) c[i] = (DT)((int32_t)(_xs() % 2001) - 1000);
#endif
    }
    if (A_IS_MATRIX) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (SYMMETRIC_A && j > i) a[(size_t)i*n + j] = a[(size_t)j*n + i];
                if (LOWER_TRIANGULAR_A && j > i) a[(size_t)i*n + j] = (DT)0;
                if (BANDED_A_RADIUS >= 0 && (j - i > BANDED_A_RADIUS || i - j > BANDED_A_RADIUS))
                    a[(size_t)i*n + j] = (DT)0;
            }
        }
    }

    agentvec_kernel(a, b, c, out, n);

    FILE *f = fopen(outp, "wb");
    if (!f) { perror("fopen"); return 3; }
    fwrite(out, sizeof(DT), (size_t)outlen, f);
    fclose(f);
    return 0;
}
