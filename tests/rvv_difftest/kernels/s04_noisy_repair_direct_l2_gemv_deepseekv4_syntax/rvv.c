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
    for (int i = 0; i < n; i++) {
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));
        const float *row_a = a + (size_t)i * n;
        
        for (int j = 0; j < n; ) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(n - j));
            vfloat32m1_t va = __riscv_vle32_v_f32m1(row_a + j, vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j, vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
            vacc = __riscv_vfadd_vv_f32m1(vacc, vprod, vl);
            j += (int)vl;
        }
        
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        sum = __riscv_vfredusum_vs_f32m1_f32m1(vacc, sum, __riscv_vsetvl_e32m1(n));
        out[i] = __riscv_vfmv_f_s_f32m1_f32(sum);
    }
}
#include "harness.h"
