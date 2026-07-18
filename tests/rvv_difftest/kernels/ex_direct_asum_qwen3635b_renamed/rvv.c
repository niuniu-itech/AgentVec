#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    float sum = 0.0f;
    
    // Use strip-mining with vsetvl for vector-length agnostic code
    for (int i = 0; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        
        // Load vector of floats
        vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
        
        // Compute absolute value of each element
        va = vfabs_v_f32m1(va, vl);
        
        // Reduce sum using reduction intrinsic
        float partial_sum = vfredosum_vs_f32m1_f32_m1(va, __riscv_vfmv_f_s_f32_m1(0.0f), vl);
        
        sum += partial_sum;
    }
    
    out[0] = sum;
}
#include "harness.h"
