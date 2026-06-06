#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vsq = __riscv_vfmul_vv_f32m1(va, va, vl);
        vsum = __riscv_vfredusum_vs_f32m1_f32m1(vsq, vsum, vl);
        
        a += vl;
        n -= vl;
    }
    
    float s = __riscv_vfmv_f_s_f32m1_f32(vsum);
    out[0] = sqrtf(s);
}
#include "harness.h"
