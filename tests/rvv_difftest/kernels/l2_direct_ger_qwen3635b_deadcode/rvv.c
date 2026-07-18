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
    // This is A + 2*x*y^T where x=a, y=b
    
    // We process row by row. For each row i:
    // out_row[i] = c_row[i] + 2.0f * a[i] * b
    
    // Since we need to be vector-length agnostic, we use strip mining.
    // For each row i, we vectorize over the n columns.
    
    for (int i = 0; i < n; i++) {
        float a_i = a[i];
        float scale = 2.0f * a_i;
        
        const float *c_row = c + i * n;
        float *out_row = out + i * n;
        
        // Strip-mine over the row
        int remaining = n;
        int offset = 0;
        
        while (remaining > 0) {
            vint32m1_t vl = __riscv_vsetvl_e32m1(remaining);
            
            // Load c_row[offset:offset+vl]
            vfloat32m1_t vc = __riscv_vle_v_f32m1(c_row + offset, vl);
            
            // Load b[offset:offset+vl]
            vfloat32m1_t vb = __riscv_vle_v_f32m1(b + offset, vl);
            
            // Compute: vc + scale * vb
            vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(scale, vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vscale, vb, vl);
            vfloat32m1_t vresult = __riscv_vfadd_vv_f32m1(vc, vprod, vl);
            
            // Store result
            __riscv_vse_v_f32m1(out_row + offset, vresult, vl);
            
            offset += vl;
            remaining -= vl;
        }
    }
}
#include "harness.h"
