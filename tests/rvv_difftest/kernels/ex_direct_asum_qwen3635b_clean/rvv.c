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
    vfloat32m1_t va;
    vfloat32m1_t vabs;
    vfloat32m1_t vse;
    
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        va = __riscv_vle_v_f32m1(a + i, vl);
        vabs = __riscv_vfsgnjn_v_f32m1(va, va);
        vse = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vse = __riscv_vfadd_vv_f32m1(vse, vabs, vl);
        s += __riscv_vfredusum_vs_f32m1_f32m1(vse, vse, vl);
        
        i += vl;
    }
    
    out[0] = s;
}
#include "harness.h"
