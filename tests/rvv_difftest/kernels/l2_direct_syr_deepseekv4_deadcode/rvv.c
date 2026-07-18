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
    // Strip-mine over rows of the output matrix
    for (int i = 0; i < n; ++i) {
        float ai = a[i];
        // Process columns in vector-length-agnostic chunks
        int j = 0;
        while (j < n) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            // Load c[i*n + j : i*n + j + vl]
            vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i * n + j], vl);
            // Load a[j : j + vl]
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[j], vl);
            // Compute 2.0f * ai * a[j]
            vfloat32m1_t vec_term = __riscv_vfmul_vf_f32m1(vec_a, 2.0f * ai, vl);
            // Add to c element
            vfloat32m1_t vec_out = __riscv_vfadd_vv_f32m1(vec_c, vec_term, vl);
            // Store result
            __riscv_vse32_v_f32m1(&out[i * n + j], vec_out, vl);
            j += vl;
        }
    }
}
#include "harness.h"
