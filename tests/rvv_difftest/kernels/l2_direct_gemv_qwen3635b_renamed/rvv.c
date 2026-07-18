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
    (void)c;
    int vl = __riscv_vsetvl_e32m1(n);
    for (int i = 0; i < n; i += vl) {
        int current_vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t sum = vmv_v_f32m1(0.0f);
        for (int j = 0; j < n; j += current_vl) {
            int inner_vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t b_vec = vle32_v_f32m1(b + j, inner_vl);
            // For each row i+k, we need a[i+k][j:j+inner_vl]
            for (int k = 0; k < current_vl; k++) {
                if (i + k < n) {
                    vfloat32m1_t a_row = vle32_v_f32m1(a + (i + k) * n + j, inner_vl);
                    vfloat32m1_t prod = vmul_vv_f32m1(a_row, b_vec, inner_vl);
                    sum = vadd_vv_f32m1(sum, prod, inner_vl);
                }
            }
        }
        vse32_v_f32m1(out + i, sum, current_vl);
    }
}
#include "harness.h"
