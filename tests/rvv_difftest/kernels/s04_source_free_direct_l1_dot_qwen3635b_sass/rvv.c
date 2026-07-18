#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl;
    vfloat32m1_t va, vb, vc, vout;
    vfloat32m1_t vsum = vmv_v_f32_m1(0.0f);
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        va = vle32_v_f32m1(a + i, vl);
        vb = vle32_v_f32m1(b + i, vl);
        vc = vle32_v_f32m1(c + i, vl);
        
        vfloat32m1_t prod = vmul_vv_f32m1(va, vb, vl);
        vfloat32m1_t sum = vadd_vv_f32m1(prod, vc, vl);
        
        vsum = vadd_vv_f32m1(vsum, sum, vl);
        
        i += vl;
    }
    
    // Reduce the accumulated vector sum to a scalar
    float result = vfredosum_vs_f32m1_f32_m1(vsum, vzero, vl, vl);
    
    // Store the result to out[0]
    out[0] = result;
}
#include "harness.h"
