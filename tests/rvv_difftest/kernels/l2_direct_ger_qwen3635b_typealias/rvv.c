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
    // out[i*n+j] = c[i*n+j] + 2.0f * a[i] * b[j]
    // This is A + 2*x*y^T where x=a, y=b, A=c
    
    // Precompute 2.0f as a vector
    vfloat32m1_t two = vmv_v_f32_m1(2.0f);
    
    int i = 0;
    int vl;
    
    // Process row by row
    for (i = 0; i < n; i++) {
        const float *c_row = c + i * n;
        float *out_row = out + i * n;
        float ai = a[i];
        
        // Compute 2.0f * a[i]
        vfloat32m1_t scaled_a = vmul_vf_f32_m1(two, ai, n);
        
        // For each element j in the row: out_row[j] = c_row[j] + scaled_a * b[j]
        // But scaled_a is a scalar value replicated, so we need to multiply by b[j]
        // Actually, scaled_a = 2.0f * a[i] for all lanes, so we can do:
        // result[j] = c_row[j] + scaled_a * b[j]
        
        int j = 0;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            
            vfloat32m1_t vc = vle_v_f32_m1(c_row + j, vl);
            vfloat32m1_t vb = vle_v_f32_m1(b + j, vl);
            
            // Multiply scaled_a (scalar) by vb (vector)
            vfloat32m1_t vprod = vmul_vf_f32_m1(vb, ai * 2.0f, vl);
            
            // Add to c
            vfloat32m1_t vout = vadd_vv_f32_m1(vc, vprod, vl);
            
            vse_v_f32_m1(vout, out_row + j, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
