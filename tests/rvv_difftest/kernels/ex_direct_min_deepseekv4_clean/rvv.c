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
    int i = 1;
    for (; i + __riscv_vsetvl_e32m1(n - i) <= n; ) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t min_vec = __riscv_vfmv_v_f_f32m1(s, vl);
        vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(vec, min_vec, vl);
        s = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vec, __riscv_vfmv_v_f_f32m1(s, vl), mask, vl));
        i += vl;
    }
    if (i < n) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t min_vec = __riscv_vfmv_v_f_f32m1(s, vl);
        vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(vec, min_vec, vl);
        s = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vec, __riscv_vfmv_v_f_f32m1(s, vl), mask, vl));
    }
    out[0] = s;
}
#include "harness.h"
