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
    
    // Preload a into a vector register for reuse across columns
    // We'll process row by row. For each row i, we compute:
    // out[i*n + j] = c[i*n + j] + 2.0f * a[i] * a[j] for j = 0..n-1
    
    // Since n can be arbitrary, we use strip-mining with vsetvl
    
    for (int i = 0; i < n; i++) {
        float a_i = a[i];
        float scale = 2.0f * a_i;
        
        // Process the row i: out[i*n + j] = c[i*n + j] + scale * a[j]
        int remaining = n;
        int offset = i * n;
        
        while (remaining > 0) {
            // Determine the vector length for this chunk
            size_t vl = __riscv_vsetvl_e32m1(remaining);
            
            // Load vector of a[j] for the current chunk
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a + offset, vl);
            
            // Load vector of c[i*n + j] for the current chunk
            vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + offset, vl);
            
            // Multiply each a[j] by scale
            vfloat32m1_t vs = __riscv_vfmul_vf_f32m1(va, scale, vl);
            
            // Add to c
            vfloat32m1_t vout = __riscv_vadd_vf_f32m1(vc, vs, vl);
            
            // Store result
            __riscv_vse32_v_f32m1(out + offset, vout, vl);
            
            // Advance pointers and remaining count
            a += vl;
            c += vl;
            out += vl;
            remaining -= vl;
        }
    }
}
#include "harness.h"
