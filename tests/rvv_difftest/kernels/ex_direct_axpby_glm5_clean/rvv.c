#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int i = 0;
    while (i < n) {
        vuint32m1_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t v2a = __riscv_vfmul_vf_f32m1(va, 2.0f, vl);
        vfloat32m1_t v3b = __riscv_vfmul_vf_f32m1(vb, 3.0f, vl);
        vfloat32m1_t vout = __riscv_vfadd_vv_f32m1(v2a, v3b, vl);
        __riscv_vse32_v_f32m1(out + i, vout, vl);
        i += vl;
    }
}
#include "harness.h"
