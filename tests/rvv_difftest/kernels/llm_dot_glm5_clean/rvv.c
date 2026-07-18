#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    float s = 0;
    size_t vl;
    vfloat32m1_t v_s = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    while (n > 0) {
        vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t v_b = __riscv_vle32_v_f32m1(b, vl);
        vfloat32m1_t v_prod = __riscv_vfmul_vv_f32m1(v_a, v_b, vl);
        v_s = __riscv_vfredusum_vs_f32m1_f32m1(v_prod, v_s, vl);
        a += vl;
        b += vl;
        n -= vl;
    }

    s = __riscv_vfmv_f_s_f32m1_f32(v_s);
    out[0] = s;
}
#include "harness.h"
