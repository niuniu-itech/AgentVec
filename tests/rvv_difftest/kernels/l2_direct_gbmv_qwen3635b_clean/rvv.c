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
        // Determine the band for row i: j from max(0, i-2) to min(n-1, i+2)
        int lo = i - 2;
        if (lo < 0) lo = 0;
        int hi = i + 2;
        if (hi >= n) hi = n - 1;
        
        int num_elements = hi - lo + 1;
        
        // For small bands, scalar is fine and simpler
        if (num_elements <= 4) {
            float s = 0.0f;
            for (int j = lo; j <= hi; j++) {
                s += a[i * n + j] * b[j];
            }
            out[i] = s;
        } else {
            // Use vector operations for larger bands (though max band is 5, so this path may not be taken)
            // Actually max band size is 5 (when i >= 2 and i <= n-3), so we can handle it with vsetvl
            // Let's use a general approach that works for any band size up to 5
            
            // We'll use a vectorized approach that handles the variable-length band
            vfloat32m1_t vs = vmv_v_f_f32m1_m(0.0f, 0.0f); // Initialize sum to 0
            
            // Process the band elements
            int remaining = num_elements;
            int j = lo;
            
            while (remaining > 0) {
                // Determine how many elements we can load in this iteration
                int vl = __riscv_vsetvl_e32m1(remaining);
                
                // Load b[j:j+vl]
                vfloat32m1_t vb = vle32_v_f32m1(b + j, vl);
                
                // Load a[i*n+j:i*n+j+vl]
                vfloat32m1_t va = vle32_v_f32m1(a + i * n + j, vl);
                
                // Multiply and accumulate
                vs = vfmacc_vv_f32m1_m(1.0f, va, vb, vs);
                
                j += vl;
                remaining -= vl;
            }
            
            // Extract the result
            float result;
            vse32_v_f32m1(&result, vs, 1);
            out[i] = result;
        }
    }
}
#include "harness.h"
