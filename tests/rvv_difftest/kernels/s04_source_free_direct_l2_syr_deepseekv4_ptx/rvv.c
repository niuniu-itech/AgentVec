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
    int total = n * n;
    int vl;
    for (int i = 0; i < total; i += vl) {
        vl = __riscv_vsetvl_e32m1(total - i);
        vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i], vl);
        int row = i / n;
        int col = i % n;
        float xi = a[row];
        float xj = b[col];
        vfloat32m1_t vec_term = __riscv_vfmul_vf_f32m1(__riscv_vfmul_vf_f32m1(vec_c, 2.0f, vl), xi * xj, vl);
        __riscv_vse32_v_f32m1(&out[i], vec_term, vl);
    }
}
#include "harness.h"
