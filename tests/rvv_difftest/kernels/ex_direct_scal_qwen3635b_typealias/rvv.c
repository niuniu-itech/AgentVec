#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    int vl;
    int i = 0;
    
    // Process in chunks using strip-mining
    for (i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        // Load vl elements from a
        vfloat32m1_t va = __riscv_vle_v_f32m1(a + i, vl);
        
        // Multiply by 2.0f
        vfloat32m1_t result = __riscv_vfmul_vf_f32m1(va, 2.0f, vl);
        
        // Store vl elements to out
        __riscv_vse_v_f32m1(out + i, result, vl);
    }
}
#include "harness.h"
