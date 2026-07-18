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
    int vl;
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_mul = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            acc = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vec_mul, __riscv_vfmv_s_f_f32m1(acc, vl), vl));
        }
        out[i] = acc;
    }
}
#include "harness.h"
