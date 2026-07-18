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
    
    // Use strip-mining with vsetvl to handle arbitrary n in a vector-length agnostic way
    for (int i = 0; i < n; ) {
        vltype_t vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vl elements from a
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        
        // Compute square of each element: va * va
        vfloat32m1_t va_sq = __riscv_vmul_vv_f32m1(va, va, vl);
        
        // Reduce sum of squares using vfredosum
        float partial_sum = __riscv_vfredosum_vs_f32m1_f32_m1(va_sq, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        
        sum += partial_sum;
        
        i += vl;
    }
    
    out[0] = sqrtf(sum);
}
#include "harness.h"
