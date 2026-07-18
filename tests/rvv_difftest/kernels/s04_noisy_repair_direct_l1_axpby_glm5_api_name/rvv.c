#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v0 = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t v1 = __riscv_vfmul_vf_f32m1(v0, 2.0f, vl);
        vfloat32m1_t v2 = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t v3 = __riscv_vfmul_vf_f32m1(v2, 3.0f, vl);
        vfloat32m1_t v4 = __riscv_vfadd_vv_f32m1(v1, v3, vl);
        __riscv_vse32_v_f32m1(out + i, v4, vl);
    }
}
#include "harness.h"
