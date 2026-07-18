#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl;
    vfloat32m1_t va, vb, vc, vout1, vout2;
    vuint32m1_t vidx;
    
    // Note: 'c' is unused in the provided assembly evidence, but kept in signature.
    // The assembly shows: out[i] = b[i], out[n+i] = a[i]
    
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        // Load vectors
        va = __riscv_vle32_v_f32m1(a + i, vl);
        vb = __riscv_vle32_v_f32m1(b + i, vl);
        
        // Store results
        // out[i] = b[i]
        __riscv_vse32_v_f32m1(out + i, vb, vl);
        
        // out[n+i] = a[i]
        __riscv_vse32_v_f32m1(out + n + i, va, vl);
    }
}
#include "harness.h"
