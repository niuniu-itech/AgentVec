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
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(3.0e38f, vlmax);
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfmin_vv_f32m1(acc, v, vl);
    }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(3.0e38f, 1);
    r = __riscv_vfredmin_vs_f32m1_f32m1(acc, r, vlmax);
    __riscv_vse32_v_f32m1(out, r, 1);
}
#include "harness.h"
