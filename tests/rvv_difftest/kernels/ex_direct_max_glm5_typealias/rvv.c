#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;

    if (n <= 0) {
        return;
    }

    size_t vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t v_s = __riscv_vle32_v_f32m1(a, vl);

    int i = 1;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(&a[i], vl);
        v_s = __riscv_vfmax_vv_f32m1(v_s, v_a, vl);
        i += vl;
    }

    vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t v_red = __riscv_vfmv_v_f_f32m1(-__builtin_inff(), vl);
    v_red = __riscv_vfredmax_vs_f32m1_f32m1(v_s, v_red, vl);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(v_red);
}
#include "harness.h"
