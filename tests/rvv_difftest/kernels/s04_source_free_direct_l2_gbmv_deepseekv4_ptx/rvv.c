#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS 2
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        int start = (i - 2 > 0) ? i - 2 : 0;
        int end = (i + 2 < n) ? i + 2 : n - 1;
        for (int j = start; j <= end; j++) {
            acc += a[i * n + j] * b[j];
        }
        out[i] = acc;
    }
}
#include "harness.h"
