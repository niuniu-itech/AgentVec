#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    
    float s = 0.0f;
    
    // Strip-mining loop for vector-length agnostic code
    for (int i = 0; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vectors
        vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b + i, vl);
        
        // Multiply and accumulate
        vfloat32m1_t vmul = vmul_vv_f32m1(va, vb, vl);
        
        // Reduce sum using vreduce_add_vf
        float partial_sum = vreduce_add_vf_f32_m1(s, vmul, vl);
        s = partial_sum;
    }
    
    out[0] = s;
}
#include "harness.h"
