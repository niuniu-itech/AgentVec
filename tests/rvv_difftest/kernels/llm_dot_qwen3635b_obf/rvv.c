#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    
    float sum = 0.0f;
    
    // Strip-mining loop for vector-length-agnostic execution
    for (int i = 0; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        
        if (vl > 0) {
            vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
            vfloat32m1_t vb = vle32_v_f32m1(b + i, vl);
            
            vfloat32m1_t vmul = vmul_vv_f32m1(va, vb, vl);
            
            // Reduce the vector to a scalar sum
            // vreduce_add_vf_f32m1_f32 reduces the vector to a single float
            float partial_sum = vreduce_add_vf_f32m1_f32(vmul);
            
            sum += partial_sum;
        }
    }
    
    out[0] = sum;
}
#include "harness.h"
