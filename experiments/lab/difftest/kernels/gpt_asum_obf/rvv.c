#include <riscv_vector.h>
#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b; (void)c;
    size_t vlmax = __riscv_vsetvl_e32m1((size_t)-1);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        v = __riscv_vfabs_v_f32m1(v, vl);
        acc = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl);
        i += (int)vl;
    }
    out[0] = __riscv_vfmv_f_s_f32m1_f32(acc);
}
#include "harness.h"
