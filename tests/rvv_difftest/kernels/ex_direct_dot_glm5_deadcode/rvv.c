#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;

    float s = 0.0f;
    int i = 0;
    size_t vlmax = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    for (; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vsum = __riscv_vfadd_vv_f32m1(vsum, vprod, vl);
        i += vl;
    }

    // Horizontal sum of the vector accumulator
    size_t vl = __riscv_vsetvl_e32m1(vlmax);
    vfloat32m1_t vshift1 = __riscv_vslidedown_vx_f32m1(vsum, 1, vl);
    vsum = __riscv_vfadd_vv_f32m1(vsum, vshift1, vl);
    vl = __riscv_vsetvl_e32m1(vlmax);
    vfloat32m1_t vshift2 = __riscv_vslidedown_vx_f32m1(vsum, 2, vl);
    vsum = __riscv_vfadd_vv_f32m1(vsum, vshift2, vl);

    s = __riscv_vfmv_f_s_f32m1_f32(vsum);

    out[0] = s;
}
#include "harness.h"
