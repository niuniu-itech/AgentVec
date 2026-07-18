#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vmul = __riscv_vfmul_vf_f32m1(va, 2.0f, vl);
        vfloat32m1_t vadd = __riscv_vfadd_vv_f32m1(vmul, vb, vl);
        __riscv_vse32_v_f32m1(out + i, vadd, vl);
        i += vl;
    }
}
#include "harness.h"
