#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c; size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax); size_t vl;
    for (int i = 0; i < n; i += (int)vl) { vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, va, va, vl); }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    out[0] = sqrtf(__riscv_vfmv_f_s_f32m1_f32(r));
}
#include "harness.h"
