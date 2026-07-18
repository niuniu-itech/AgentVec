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
    
    // Precompute 2.0f as a vector
    vfloat32m1_t two = vfmv_v_f_f32m1(2.0f, 1);
    
    // Process in strip-mined chunks
    int i = 0;
    for (; i < n; i += 1) {
        // For each row i, we compute out[i*n + j] = c[i*n + j] + 2.0f * a[i] * a[j] for j=0..n-1
        
        // Load a[i] scalar
        float a_i = a[i];
        vfloat32m1_t a_i_vec = vfmv_v_f_f32m1(a_i, 1);
        
        // Precompute 2.0f * a[i] as a vector
        vfloat32m1_t two_a_i = vfmul_vf_f32m1(two, a_i, 1);
        
        // Process columns j in vector chunks
        int j = 0;
        for (; j < n; j += 1) {
            // For each column j, we need a[j]
            float a_j = a[j];
            vfloat32m1_t a_j_vec = vfmv_v_f_f32m1(a_j, 1);
            
            // Compute 2.0f * a[i] * a[j] as a scalar
            float correction = two_a_i[0] * a_j;
            
            // Load the row of c starting at c[i*n + j]
            // We need to load n elements from c[i*n + j] but we're processing one j at a time
            // Actually, let's restructure: for each row i, process all j in vector chunks
            
            // This approach is inefficient. Let me restructure properly.
        }
    }
    
    // Let me rewrite with proper strip-mining over the entire n*n output
    // out[i*n + j] = c[i*n + j] + 2.0f * a[i] * a[j]
    
    // For each row i:
    //   For each j in 0..n-1:
    //     out[i*n + j] = c[i*n + j] + 2.0f * a[i] * a[j]
    
    // We can vectorize over j for each fixed i, using vsetvl for the vector length
    
    for (int i = 0; i < n; i++) {
        float a_i = a[i];
        vfloat32m1_t a_i_vec = vfmv_v_f_f32m1(a_i, 1);
        vfloat32m1_t two_a_i = vfmul_vf_f32m1(two, a_i, 1);
        
        int j = 0;
        int vl;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load vl elements from c[i*n + j]
            vfloat32m1_t c_vec = vle32_v_f32m1(c + i * n + j, vl);
            
            // Load vl elements from a[j] (contiguous in memory since a is a vector)
            vfloat32m1_t a_j_vec = vle32_v_f32m1(a + j, vl);
            
            // Compute correction = 2.0f * a[i] * a[j] for each element
            // two_a_i is a scalar broadcast, a_j_vec is the vector
            vfloat32m1_t correction = vfmul_vf_f32m1(two_a_i, a_j_vec, vl);
            
            // out = c + correction
            vfloat32m1_t out_vec = vadd_vv_f32m1(c_vec, correction, vl);
            
            // Store result
            vse32_v_f32m1(out + i * n + j, out_vec, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
