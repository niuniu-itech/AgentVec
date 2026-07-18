#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    size_t vl;
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (size_t i = 0; i < (size_t)n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vout = __riscv_vfmul_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(out + i, vout, vl);
    }
}
#include "harness.h"
