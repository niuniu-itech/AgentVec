#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        
        vfloat32m1_t vresult = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(va, 2.0f, vl), 3.0f, vb, vl);
        
        __riscv_vse32_v_f32m1(out + i, vresult, vl);
        i += vl;
    }
}
#include "harness.h"
