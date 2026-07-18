#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float acc = __builtin_inff();
    int i = 0;
    while (i < n) {
        long vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(acc, vl);
        vacc = __riscv_vfmin_vv_f32m1(vacc, va, vl);
        vfloat32m1_t red = __riscv_vfredmin_vs_f32m1_f32m1(vacc, __riscv_vfmv_v_f_f32m1(__builtin_inff(), vl), vl);
        acc = __riscv_vfmv_f_s_f32m1_f32(red);
        i += vl;
    }
    out[0] = acc;
}
#include "harness.h"
