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
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    size_t vl;
    for (size_t i = 0; i < (size_t)n; i++) {
        float acc = 0.0f;
        size_t j = 0;
        while (j < (size_t)n) {
            vl = vsetvl_e32m1((size_t)n - j);
            vfloat32m1_t vec_a = vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vec_b = vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_prod = vfmul_vv_f32m1(vec_a, vec_b, vl);
            acc = vfmv_f_s_f32m1_f32(vfredusum_vs_f32m1_f32m1(vec_prod, vfmv_s_f_f32m1_f32(vfloat32m1(), acc, 1), vl));
            j += vl;
        }
        out[i] = acc;
    }
}
#include "harness.h"
