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
    
    float s = 0.0f;
    int vl;
    int i = 0;
    
    // Process in chunks using strip-mining
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vector of floats
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        
        // Compute absolute value
        va = __riscv_vfabs_v_f32m1(va, vl);
        
        // Reduce sum using reduction intrinsic
        float chunk_sum = __riscv_vfredosumsu_f32_m1_f32(0.0f, va, vl);
        
        s += chunk_sum;
        i += vl;
    }
    
    out[0] = s;
}
#include "harness.h"
