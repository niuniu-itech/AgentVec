#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>
#include <stdint.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + i, vl);
        vfloat32m1_t v = __riscv_vfmul_vv_f32m1(va, vb, vl);
        v = __riscv_vfadd_vv_f32m1(v, vc, vl);
        acc = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl);
    }
    float sum = __riscv_vfredusum_vs_f32m1_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.0f, vlmax), vlmax);
    *out = sqrtf(sum);
}
#include "harness.h"
