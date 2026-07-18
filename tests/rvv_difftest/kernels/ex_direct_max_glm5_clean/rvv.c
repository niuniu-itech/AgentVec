#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;

    float s = a[0];
    size_t vl = __riscv_vsetvl_e32m1(n - 1);
    vfloat32m1_t v_s = __riscv_vfmv_v_f_f32m1(s, vl);

    size_t i = 1;
    for (; i < n;) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vbool32_t mask = __riscv_vmfgt_vv_f32m1_b32(v_a, v_s, vl);
        v_s = __riscv_vmerge_vvm_f32m1(v_s, v_a, mask, vl);
        i += vl;
    }

    // Horizontal max reduction
    vl = __riscv_vsetvl_e32m1(1);
    v_s = __riscv_vfredmax_vs_f32m1_f32m1(v_s, __riscv_vfmv_v_f_f32m1(-__builtin_inff(), vl), vl);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(v_s);
}
#include "harness.h"
