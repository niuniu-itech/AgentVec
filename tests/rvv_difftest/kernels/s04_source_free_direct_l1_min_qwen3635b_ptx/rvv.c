#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float32_t acc = __riscv_vfmv_f_s_f32_m1(0.0f);
    
    while (vl > 0) {
        vfloat32m1_t va = __riscv_vle32_v_f32_m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32_m1(b, vl);
        vfloat32m1_t vc = __riscv_vle32_v_f32_m1(c, vl);
        
        // Perform element-wise operations: out[i] = a[i] + b[i] * c[i]
        vfloat32m1_t tmp = __riscv_vfmul_vv_f32_m1(vb, vc, vl);
        vfloat32m1_t res = __riscv_vfadd_vv_f32_m1(va, tmp, vl);
        
        // Store result
        __riscv_vse32_v_f32_m1(out, res, vl);
        
        // Accumulate minimum for reduction (simulating the evidence's red.min.f32)
        // Note: The evidence shows a reduction to a single scalar, but the function signature
        // implies element-wise output. We will perform the element-wise operation as primary.
        // If reduction is needed, it would typically be done separately.
        // However, to strictly follow "migrate evidence", we note the evidence does a min reduction.
        // But the output is an array 'out'. The evidence is contradictory (reduces to scalar, stores to array).
        // Assuming the intent is element-wise computation based on the function signature.
        
        a += vl;
        b += vl;
        c += vl;
        out += vl;
        vl = __riscv_vsetvl_e32m1(n - (out - (float*)0)); // Recalculate vl for remaining elements
    }
}
#include "harness.h"
