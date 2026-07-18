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
    vfloat32m1_t va, vsq, vsum;
    vfloat32m1_t vzero = vmv_v_f_f32m1(0.0f);
    
    // Initialize accumulator
    vsum = vzero;
    
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vector from a
        va = vle32_v_f32m1(&a[i], vl);
        
        // Square each element
        vsq = vmul_vv_f32m1(va, va, vl);
        
        // Accumulate into vsum
        vsum = vadd_vv_f32m1(vsum, vsq, vl);
        
        i += vl;
    }
    
    // Reduce vsum to scalar sum
    // Use reduction intrinsic to sum all elements in vsum
    s = __riscv_vfredosum_vs_f32m1_f32(vsum, vzero, vl, vl);
    
    // Compute square root
    out[0] = sqrtf(s);
}
#include "harness.h"
