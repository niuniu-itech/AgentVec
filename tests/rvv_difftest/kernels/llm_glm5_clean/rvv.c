#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vfloat32m1_t vres = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vf_f32m1(va, 2.0f, vl), vb, vl);
        __riscv_vse32_v_f32m1(out, vres, vl);
        a += vl;
        b += vl;
        out += vl;
        n -= vl;
    }
}
#include "harness.h"
