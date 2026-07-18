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
        // Broadcast a[i] to all vector lanes
        float ai = a[i];
        
        // Process columns in vector-length agnostic chunks
        int j = 0;
        while (j < n) {
            // Set vector length for this iteration
            size_t vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load b[j:j+vl] and c[i*n+j:i*n+j+vl]
            vfloat32m1_t vb = __riscv_vlse32_v_f32m1(&b[j], sizeof(float), vl);
            vfloat32m1_t vc = __riscv_vlse32_v_f32m1(&c[i * n + j], sizeof(float), vl);
            
            // Compute 2.0f * a[i] * b[j] + c[i*n+j]
            vfloat32m1_t vprod = __riscv_vfmul_vf_f32m1(vb, 2.0f * ai, vl);
            vfloat32m1_t vout = __riscv_vfadd_vv_f32m1(vc, vprod, vl);
            
            // Store result to out[i*n+j:i*n+j+vl]
            __riscv_vsse32_v_f32m1(&out[i * n + j], vout, sizeof(float), vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
