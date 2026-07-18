#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 1
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        int j = 0;
        int row_start = i * n;
        while (j <= i) {
            int vl = __riscv_vsetvl_e32m1(i - j + 1);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[row_start + j], vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_prod = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_prod, vec_sum, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(vec_sum);
            j += vl;
        }
        out[i] = sum;
    }
}
#include "harness.h"
