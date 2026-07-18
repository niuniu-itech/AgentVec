#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    
    float s = 0.0f;
    int vl;
    vfloat32m1_t va, vb, vprod;
    vfloat32m1_t vsum = vmv_v_f_f32_m1(0.0f);
    
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl <= 0) break;
        
        va = vle32_v_f32_m1(a + i, vl);
        vb = vle32_v_f32_m1(b + i, vl);
        vprod = vmul_vv_f32_m1(va, vb, vl);
        vsum = vfredosum_vs_f32_m1_f32_m1(vsum, vprod, vsum, vl);
        
        i += vl;
    }
    
    s = vfredosum_vs_f32_m1_f32_m1(vsum, vsum, vsum, 0);
    out[0] = s;
}
#include "harness.h"
