#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + i, vl);
        vfloat32m1_t vout = __riscv_vfmul_vf_f32m1(va, 2.0f, vl);
        vfloat32m1_t vsum = __riscv_vfadd_vv_f32m1(vout, vb, vl);
        vfloat32m1_t vfinal = __riscv_vfadd_vv_f32m1(vsum, vc, vl);
        __riscv_vse32_v_f32m1(out + i, vfinal, vl);
    }
}
#include "harness.h"
