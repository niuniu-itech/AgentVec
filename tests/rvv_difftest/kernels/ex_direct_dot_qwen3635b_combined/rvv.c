#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    
    float result = 0.0f;
    
    if (n <= 0) {
        out[0] = result;
        return;
    }
    
    // Use strip-mining with vsetvl for vector-length agnostic code
    size_t vl;
    float acc = 0.0f;
    
    // We need to compute dot product: sum(a[i] * b[i]) for i=0..n-1
    // Using vfmacc.vv or vfmul.vv + vfadd.vv approach
    
    // Process in chunks using strip mining
    size_t remaining = n;
    const float *a_ptr = a;
    const float *b_ptr = b;
    
    while (remaining > 0) {
        vl = __riscv_vsetvl_e32m1(remaining);
        
        if (vl == 0) break;
        
        // Load vectors
        vfloat32m1_t va = __riscv_vle_v_f32m1(a_ptr, vl);
        vfloat32m1_t vb = __riscv_vle_v_f32m1(b_ptr, vl);
        
        // Multiply and accumulate
        vfloat32m1_t vmul = __riscv_vmul_vv_f32m1(va, vb, vl);
        
        // Reduce the vector to a scalar by summing
        // Use vfredsum.vs to reduce the vector to a single scalar
        float partial = 0.0f;
        vfloat32m1_t vpartial = __riscv_vfmv_v_f_f32m1(partial, vl);
        vpartial = __riscv_vfredsum_vs_f32m1_f32m1(vmul, vpartial, vl);
        partial = __riscv_vmv_x_x_f32m1_f32(vpartial);
        
        acc += partial;
        
        a_ptr += vl;
        b_ptr += vl;
        remaining -= vl;
    }
    
    out[0] = acc;
}
#include "harness.h"
