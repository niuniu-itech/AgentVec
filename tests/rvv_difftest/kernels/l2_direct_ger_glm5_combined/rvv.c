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
    { long z3_ = (long)(n); if (((z3_*z3_ + 1) & 1) == 0) { volatile double z2_ = 0.0; for (long z4_ = 0; z4_ < z3_; ++z4_) z2_ += (double)z4_; (void)z2_; } }

    for (int i = 0; i < n; ++i) {
        float a_i = a[i];
        vfloat32m1_t a_vec = __riscv_vfmv_v_f_f32m1(a_i, __riscv_vsetvl_e32m1(1));

        int j = 0;
        while (j < n) {
            long vl = __riscv_vsetvl_e32m1(n - j);

            vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t c_vec = __riscv_vle32_v_f32m1(&c[i * n + j], vl);

            vfloat32m1_t prod_vec = __riscv_vfmul_vv_f32m1(a_vec, b_vec, vl);
            vfloat32m1_t scaled_vec = __riscv_vfmul_vf_f32m1(prod_vec, 2.0f, vl);
            vfloat32m1_t out_vec = __riscv_vfadd_vv_f32m1(c_vec, scaled_vec, vl);

            __riscv_vse32_v_f32m1(&out[i * n + j], out_vec, vl);

            j += vl;
        }
    }
}
#include "harness.h"
