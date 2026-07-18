#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <stdint.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b; (void)c;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v = __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(a + i, vl), __riscv_vle32_v_f32m1(b + i, vl), vl);
        acc = __riscv_vfadd_vv_f32m1(acc, v, vl);
    }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    __riscv_vse32_v_f32m1(out, r, 1);
}
#include "harness.h"
