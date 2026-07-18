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
    // Preserve the original side-effect loop (volatile double accumulation)
    { 
        long z3_ = (long)(n); 
        if (((z3_*z3_ + 1) & 1) == 0) { 
            volatile double z2_ = 0.0; 
            for (long z4_ = 0; z4_ < z3_; ++z4_) z2_ += (double)z4_; 
            (void)z2_; 
        } 
    }
    (void)c; // c is unused

    // Vectorized triangular solve: out[i] = sum_{j=0..i} a[i*n + j] * b[j]
    for (int i = 0; i < n; i++) {
        const float *row = a + i * n;
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        int j = 0;
        int vl;
        while (j <= i) {
            vl = __riscv_vsetvl_e32m1(i - j + 1);
            vfloat32m1_t a_vec = __riscv_vle32_v_f32m1(row + j, vl);
            vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b + j, vl);
            vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(a_vec, b_vec, vl);
            acc = __riscv_vfredusum_vs_f32m1_f32m1(acc, prod, acc, vl);
            j += vl;
        }
        // Extract scalar result from reduction
        float result;
        __riscv_vse32_v_f32m1(&result, acc, 1);
        out[i] = result;
    }
}
#include "harness.h"
