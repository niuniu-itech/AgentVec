#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    
    // Process in strip-mined chunks
    for (int i = 0; i < n; i++) {
        // Determine the band limits for row i
        int lo = i - 2;
        if (lo < 0) lo = 0;
        int hi = i + 2;
        if (hi >= n) hi = n - 1;
        
        int band_len = hi - lo + 1;
        
        // Compute the starting index in b for this band
        // b[j] for j in [lo, hi]
        // a[i*n + j] for j in [lo, hi]
        
        float s = 0.0f;
        
        // Use vector operations for the band dot product
        // We'll process the band in chunks of vlen
        int idx = lo;
        while (idx <= hi) {
            // Determine how many elements we can process in this step
            int remaining = hi - idx + 1;
            // Get the vector length for this chunk
            // We use e32m1 for float32
            vint32m1_t vl = __riscv_vsetvl_e32m1(remaining);
            
            if (vl > 0) {
                // Load elements from b[idx : idx+vl]
                vfloat32m1_t vb = __riscv_vle_v_f32m1(b + idx, vl);
                
                // Load elements from a[i*n + idx : i*n + idx + vl]
                vfloat32m1_t va = __riscv_vle_v_f32m1(a + i * n + idx, vl);
                
                // Multiply and accumulate
                vfloat32m1_t vmul = __riscv_vmul_vv_f32m1(va, vb, vl);
                
                // Reduce to scalar
                s += __riscv_vfredosum_vs_f32m1_f32_m1(vmul, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
            }
            
            idx += vl;
        }
        
        out[i] = s;
    }
}
#include "harness.h"
