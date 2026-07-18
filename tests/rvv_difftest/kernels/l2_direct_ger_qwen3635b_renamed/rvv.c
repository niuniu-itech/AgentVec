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
    // a is x (n elements), b is y (n elements), c is A (n x n row-major)
    // out = c + 2 * a * b^T
    // out[i*n + j] = c[i*n + j] + 2.0f * a[i] * b[j]

    vfloat32m1_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t v2 = __riscv_vfmv_v_f_f32m1(2.0f, vl);

    for (int i = 0; i < n; i++) {
        // Load a[i] and broadcast to vector
        float a_i = a[i];
        vfloat32m1_t va = __riscv_vfmv_v_f_f32m1(a_i, vl);

        // Load b[j] for j in [0, vl)
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);

        // Compute 2.0f * a[i] * b[j]
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(v2, vb, vl);
        vprod = __riscv_vfmul_vf_f32m1(vprod, va, vl);

        // Load c[i*n + j] for j in [0, vl)
        vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + i * n, vl);

        // out[i*n + j] = c[i*n + j] + 2*a[i]*b[j]
        vfloat32m1_t vout = __riscv_vfadd_vv_f32m1(vc, vprod, vl);

        // Store result
        __riscv_vse32_v_f32m1(out + i * n, vout, vl);
    }
}
#include "harness.h"
