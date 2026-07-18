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
    vfloat32m1_t vtmp;
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl > 0) {
            va = vle32_v_f32m1(a + i, vl);
            vtmp = vmul_vv_f32m1(va, va, vl);
            s += vfredosum_vs_f32m1_f32_m1(vtmp, vzero, vl).vl_f32;
            i += vl;
        }
    }
    
    out[0] = sqrtf(s);
}
#include "harness.h"
