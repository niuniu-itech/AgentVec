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
    int n2 = n * n;
    for (int i = 0; i < n2; i += vlenb / sizeof(float)) {
        int vl = __riscv_vsetvl_e32m1(n2 - i);
        vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(c + i, vl);
        int row = i / n;
        int col = i % n;
        vfloat32m1_t vec_x = __riscv_vfmv_v_f_f32m1(*(a + row), vl);
        vfloat32m1_t vec_y = __riscv_vfmv_v_f_f32m1(*(b + col), vl);
        vfloat32m1_t term = __riscv_vfmul_vv_f32m1(vec_x, vec_y, vl);
        term = __riscv_vfadd_vv_f32m1(term, term, vl);
        vfloat32m1_t result = __riscv_vfmacc_vv_f32m1(vec_c, term, __riscv_vfmv_v_f_f32m1(2.0f, vl), vl);
        __riscv_vse32_v_f32m1(out + i, result, vl);
    }
}
#include "harness.h"
