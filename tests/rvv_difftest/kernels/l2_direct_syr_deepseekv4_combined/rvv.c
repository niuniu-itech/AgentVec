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
    // Preserve the original side-effect loop (unchanged)
    { long z3_ = (long)(n); if (((z3_*z3_ + 1) & 1) == 0) { volatile double z2_ = 0.0; for (long z4_ = 0; z4_ < z3_; ++z4_) z2_ += (double)z4_; (void)z2_; } }
    (void)b;

    // Strip-mine over rows
    for (int i = 0; i < n; i++) {
        // Broadcast a[i] to a vector
        float ai = a[i];
        vfloat32m1_t vec_ai = __riscv_vfmv_v_f_f32m1(ai, __riscv_vsetvl_e32m1(n));

        // Process columns in vector-length agnostic chunks
        int vl;
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);

            // Load row i of matrix A: c[i*n + j ...]
            vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i * n + j], vl);

            // Load a[j ... j+vl-1] (the x vector elements for this column chunk)
            vfloat32m1_t vec_a_j = __riscv_vle32_v_f32m1(&a[j], vl);

            // Compute 2 * a[i] * a[j]
            vfloat32m1_t vec_term = __riscv_vfmul_vv_f32m1(vec_ai, vec_a_j, vl);
            vec_term = __riscv_vfadd_vf_f32m1(vec_term, vec_term, vl); // multiply by 2

            // out = c + 2*x_i*x_j
            vfloat32m1_t vec_out = __riscv_vfadd_vv_f32m1(vec_c, vec_term, vl);

            // Store result
            __riscv_vse32_v_f32m1(&out[i * n + j], vec_out, vl);
        }
    }
}
#include "harness.h"
