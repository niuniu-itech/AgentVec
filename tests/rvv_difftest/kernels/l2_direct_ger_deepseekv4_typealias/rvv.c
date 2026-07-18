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
    int vl;
    for (int i = 0; i < n; i++) {
        // Broadcast a[i] to a vector
        float ai = a[i];
        vfloat32m1_t va = __riscv_vfmv_v_f_f32m1(ai, __riscv_vsetvl_e32m1(n));
        
        // Process each row j in strips
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load b[j:j+vl]
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            
            // Compute 2.0f * a[i] * b[j:j+vl]
            vfloat32m1_t vprod = __riscv_vfmul_vf_f32m1(vb, 2.0f * ai, vl);
            
            // Load c[i*n + j : i*n + j + vl]
            vfloat32m1_t vc = __riscv_vle32_v_f32m1(&c[i * n + j], vl);
            
            // Add to get out[i*n + j : i*n + j + vl]
            vfloat32m1_t vout = __riscv_vfadd_vv_f32m1(vc, vprod, vl);
            
            // Store result
            __riscv_vse32_v_f32m1(&out[i * n + j], vout, vl);
        }
    }
}
#include "harness.h"
