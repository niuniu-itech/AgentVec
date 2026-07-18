#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)a;
    
    int i = 0;
    for (; i < n; i += 1) {
        // We need to handle the swap: out[i] = b[i], out[n+i] = a[i]
        // Since we are processing one element at a time in this simple loop,
        // let's use vector operations for efficiency but ensure correctness.
        // However, the requirement is to use strip-mining with vsetvl.
        // Let's process in chunks.
    }
    
    // Let's rewrite properly with strip mining
    i = 0;
    vfloat32m1_t va, vb, vout_a, vout_b;
    int vl;
    
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load b[i:i+vl] into vb
        vb = __riscv_vle_v_f32m1(b + i, vl);
        // Load a[i:i+vl] into va
        va = __riscv_vle_v_f32m1(a + i, vl);
        
        // Store vb to out[i:i+vl]
        __riscv_vse_v_f32m1(out + i, vb, vl);
        // Store va to out[n+i:n+i+vl]
        __riscv_vse_v_f32m1(out + n + i, va, vl);
        
        i += vl;
    }
}
#include "harness.h"
