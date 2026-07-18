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
    // Process each row of the output matrix
    for (int i = 0; i < n; i++) {
        const float ai = a[i];
        // Process columns in vector-length agnostic strips
        int j = 0;
        while (j < n) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load c[i*n + j : i*n + j + vl]
            vfloat32m1_t c_vec = __riscv_vlse32_v_f32m1(&c[i*n + j], sizeof(float), vl);
            
            // Load a[j : j + vl]
            vfloat32m1_t a_vec = __riscv_vlse32_v_f32m1(&a[j], sizeof(float), vl);
            
            // Compute 2.0f * ai * a[j + k] for k = 0..vl-1
            vfloat32m1_t prod = __riscv_vfmul_vf_f32m1(a_vec, 2.0f * ai, vl);
            
            // Add to c values
            vfloat32m1_t out_vec = __riscv_vfadd_vv_f32m1(c_vec, prod, vl);
            
            // Store result
            __riscv_vsse32_v_f32m1(&out[i*n + j], sizeof(float), out_vec, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
