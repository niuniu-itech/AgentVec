#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    size_t vl;
    vfloat32m1_t va, vb, vc, vout;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        va = __riscv_vle32_v_f32m1(a + i, vl);
        vb = __riscv_vle32_v_f32m1(b + i, vl);
        vc = __riscv_vle32_v_f32m1(c + i, vl);
        vout = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vout = __riscv_vfadd_vv_f32m1(vout, vc, vl);
        __riscv_vse32_v_f32m1(out + i, vout, vl);
    }
}
#include "harness.h"
