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
    // out = c + 2 * a * b^T
    // out[i*n + j] = c[i*n + j] + 2.0f * a[i] * b[j]
    
    // Precompute 2.0f * a[i] for each row i to avoid repeated multiplication
    // We'll process row by row, vectorizing over the columns (j dimension)
    
    for (int i = 0; i < n; i++) {
        float scaled_a = 2.0f * a[i];
        
        // Vectorize over j from 0 to n-1
        int vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t va = vmv_v_f_f32_m1(scaled_a, vl);
        // We need to load b[j], multiply by scaled_a, then add to c[i*n + j]
        
        for (int j = 0; j < n; j += vl) {
            int current_vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load b[j:j+current_vl]
            vfloat32m1_t vb = vle32_v_f32m1(b + j, current_vl);
            
            // Load c[i*n + j : i*n + j + current_vl]
            vfloat32m1_t vc = vle32_v_f32m1(c + i * n + j, current_vl);
            
            // Multiply vb by scaled_a (broadcast scalar to vector)
            vfloat32m1_t vprod = vmul_vf_f32m1(vb, scaled_a, current_vl);
            
            // Add to vc
            vfloat32m1_t vresult = vadd_vv_f32m1(vc, vprod, current_vl);
            
            // Store result
            vse32_v_f32m1(out + i * n + j, vresult, current_vl);
        }
    }
}
#include "harness.h"
