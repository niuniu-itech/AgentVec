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
        const float *a_row = a + i * n;
        int j = 0;
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

        while (j < n) {
            size_t vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a_row + j, vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j, vl);
            vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, __riscv_vfmul_vv_f32m1(va, vb, vl), vl);
            j += vl;
        }

        size_t vl = __riscv_vsetvl_e32m1(1);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vsum, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        out[i] = __riscv_vfmv_f_s_f32m1_f32(vred);
    }
}
#include "harness.h"
