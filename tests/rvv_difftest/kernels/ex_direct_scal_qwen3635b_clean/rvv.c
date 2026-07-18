#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    int vl;
    vfloat32m1_t va;
    vfloat32m1_t vout;
    vfloat32m1_t vtwo = vfmv_v_f_f32m1(2.0f);
    
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        va = vle32_v_f32m1(a + i, vl);
        vout = vfmul_vv_f32m1(va, vtwo, vl);
        vse32_v_f32m1(out + i, vout, vl);
    }
}
#include "harness.h"
