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
    (void)b; // unused parameter
    int vl;
    for (int i = 0; i < n; i++) {
        // Broadcast a[i] for the outer product
        float ai = a[i];
        // Process row i of the output matrix
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            // Load row i of matrix C (row-major: c[i*n + j])
            vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i * n + j], vl);
            // Load vector a[j:j+vl] for the outer product term
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[j], vl);
            // Compute 2 * a[i] * a[j:j+vl]
            vfloat32m1_t vec_prod = __riscv_vfmul_vf_f32m1(vec_a, 2.0f * ai, vl);
            // Add to the matrix row
            vfloat32m1_t vec_out = __riscv_vfadd_vv_f32m1(vec_c, vec_prod, vl);
            // Store result
            __riscv_vse32_v_f32m1(&out[i * n + j], vec_out, vl);
        }
    }
}
#include "harness.h"
