#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float sum = 0.0f;
    size_t vl;
    vfloat32m1_t va, vsum;
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    // Process in chunks using strip-mining
    size_t i = 0;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vector
        va = vle32_v_f32m1(a + i, vl);
        
        // Multiply element-wise by itself and accumulate
        vsum = vmul_vv_f32m1(va, va, vl);
        
        // Reduce sum using reduction intrinsic
        // We need to accumulate into a scalar sum
        // Use vfredosum_d or vfredusum_d for float reduction
        // Since we're using m1, we can use vfredosum_v_f32_m1 to reduce to scalar
        
        // Actually, let's use a simpler approach: accumulate using reduction
        // vfredosum_v_f32_m1 reduces a vector to a scalar
        float chunk_sum = __riscv_vfredosum_v_f32_m1(vsum, vzero, vl);
        sum += chunk_sum;
        
        i += vl;
    }
    
    out[0] = sqrtf(sum);
}
#include "harness.h"
