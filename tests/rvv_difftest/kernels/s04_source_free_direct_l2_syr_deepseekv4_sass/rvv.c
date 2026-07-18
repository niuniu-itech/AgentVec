#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 0
#define B_IS_MATRIX 0
#define C_IS_MATRIX 1
#define OUT_IS_MATRIX 1
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i * n + j], vl);
            vfloat32m1_t vec_xj = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_xi = __riscv_vfmv_v_f_f32m1(a[i], vl);
            vfloat32m1_t vec_t = __riscv_vfmacc_vv_f32m1(vec_c, vec_xi, vec_xj, vl);
            __riscv_vse32_v_f32m1(&out[i * n + j], vec_t, vl);
        }
    }
}
#include "harness.h"
