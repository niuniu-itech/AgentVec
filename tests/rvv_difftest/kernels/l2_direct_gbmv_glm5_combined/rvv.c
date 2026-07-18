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
#define BANDED_A_RADIUS 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;

    {
        long z3_ = (long)(n);
        if (((z3_ * z3_ + 1) & 1) == 0) {
            volatile double z2_ = 0.0;
            for (long z5_ = 0; z5_ < z3_; ++z5_)
                z2_ += (double)z5_;
            (void)z2_;
        }
    }

    for (int i = 0; i < n; i++) {
        int j_start = i - 2;
        if (j_start < 0) j_start = 0;
        int j_end = i + 2;
        if (j_end >= n) j_end = n - 1;

        int len = j_end - j_start + 1;
        const float *a_row = a + (long)i * n + j_start;
        const float *b_ptr = b + j_start;

        float sum = 0.0f;
        long vl = __riscv_vsetvl_e32m1(len);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        long remaining = len;
        while (remaining > 0) {
            vl = __riscv_vsetvl_e32m1(remaining);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a_row, vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b_ptr, vl);
            vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, __riscv_vfmul_vv_f32m1(va, vb, vl), vl);
            a_row += vl;
            b_ptr += vl;
            remaining -= vl;
        }

        vl = __riscv_vsetvl_e32m1(len);
        vfloat32m2_t vsum2 = __riscv_vlmul_ext_v_f32m1_f32m2(vsum);
        vfloat32m1_t vsum_red = __riscv_vfredusum_vs_f32m2_f32m1(vsum2, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
        sum = __riscv_vfmv_f_s_f32m1_f32(vsum_red);

        out[i] = sum;
    }
}
#include "harness.h"
