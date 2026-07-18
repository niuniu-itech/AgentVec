#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    float max_val = a[0];
    
    if (n > 1) {
        size_t vl;
        vfloat32m1_t vmax;
        
        // Initialize vmax with the first element
        vmax = vfmv_v_f_f32m1(max_val);
        
        // Process remaining elements with strip-mining
        for (size_t i = 1; i < (size_t)n; i += vl) {
            vl = __riscv_vsetvl_e32m1(n - i);
            vfloat32m1_t vdata = vle32_v_f32m1(a + i, vl);
            vmax = vmax_vv_f32m1(vmax, vdata);
        }
        
        // Extract the maximum value from the vector register
        max_val = vmin_vx_f32m1(max_val, vmax);
        // Actually, we need to reduce the vector to a scalar.
        // Use vreduce_max_vf for reduction
        max_val = __riscv_vreduce_max_vf_f32m1(max_val, vmax);
    }
    
    out[0] = max_val;
}
#include "harness.h"
