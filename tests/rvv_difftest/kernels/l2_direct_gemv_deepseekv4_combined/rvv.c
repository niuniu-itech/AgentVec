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
    // Preserve the original side-effect loop (volatile double accumulation)
    {
        long z3_ = (long)n;
        if (((z3_ * z3_ + 1) & 1) == 0) {
            volatile double z2_ = 0.0;
            for (long z4_ = 0; z4_ < z3_; ++z4_)
                z2_ += (double)z4_;
            (void)z2_;
        }
    }
    (void)c; // c is unused

    // Vectorized matrix-vector multiplication: out = A * b
    for (int i = 0; i < n; ++i) {
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));
        const float *row = a + i * n;
        int vl;
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t a_vec = __riscv_vle32_v_f32m1(row + j, vl);
            vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b + j, vl);
            sum = __riscv_vfmacc_vv_f32m1(sum, a_vec, b_vec, vl);
        }
        // Reduce sum to scalar
        vfloat32m1_t red = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        // Use a reduction loop with a single vector
        float result = 0.0f;
        for (int k = 0; k < n; k += vl) {
            vl = __riscv_vsetvl_e32m1(n - k);
            vfloat32m1_t part = __riscv_vle32_v_f32m1((const float *)&sum + k, vl);
            // Accumulate using a scalar loop for simplicity (vector reduction not needed)
            for (int t = 0; t < vl; ++t) {
                result += ((float *)&part)[t];
            }
        }
        out[i] = result;
    }
}
#include "harness.h"
