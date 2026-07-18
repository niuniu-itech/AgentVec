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
    size_t vl;
    for (size_t i = 0; i < (size_t)n; i += vl) {
        vl = vsetvl_e32m1(n - i);
        vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, vl);
        for (int j = -2; j <= 2; j++) {
            size_t jj = i + j;
            if (jj >= (size_t)n) continue;
            vfloat32m1_t v_a = vlse32_v_f32m1(&a[jj * n + i], sizeof(float) * n, vl);
            vfloat32m1_t v_b = vle32_v_f32m1(&b[jj], vl);
            acc = vfmacc_vf_f32m1(acc, b[jj], v_a, vl);
        }
        vse32_v_f32m1(&out[i], acc, vl);
    }
}
#include "harness.h"
