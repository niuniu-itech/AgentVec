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
    (void)b;
    
    // out = c + 2 * a * a^T
    // out[i*n + j] = c[i*n + j] + 2.0f * a[i] * a[j]
    
    // Precompute 2.0f * a[i] for each row i to avoid repeated multiplication
    // We'll process row by row, vectorizing the inner loop over j
    
    for (int i = 0; i < n; i++) {
        float two_a_i = 2.0f * a[i];
        
        // Vectorize the inner loop: out[i*n + j] = c[i*n + j] + two_a_i * a[j]
        // We need to load a[j] and c[i*n + j] for j = 0..n-1
        
        int vl = __riscv_vsetvl_e32m1(n);
        int j = 0;
        
        while (j < n) {
            vfloat32m1_t va = vld1_v_f32m1((const float *)&a[j], vl);
            vfloat32m1_t vc = vld1_v_f32m1((const float *)&c[i * n + j], vl);
            
            // two_a_i * va: broadcast scalar to vector and multiply
            vfloat32m1_t vt = vmv_v_f_f32m1(two_a_i);
            vfloat32m1_t vprod = vmul_vv_f32m1(vt, va, vl);
            
            vfloat32m1_t vout = vadd_vv_f32m1(vc, vprod, vl);
            
            vs1_v_f32m1((float *)&out[i * n + j], vout, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
