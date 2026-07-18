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
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));

    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vsum = __riscv_vfadd_vv_f32m1(vsum, vprod, vl);
        i += vl;
    }

    size_t vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vshifted = __riscv_vslide1down_vf_f32m1(vsum, 0.0f, vl);
    vsum = __riscv_vfadd_vv_f32m1(vsum, vshifted, vl);

    vfloat32m1_t vshifted2 = __riscv_vslide1down_vf_f32m1(vsum, 0.0f, vl);
    vsum = __riscv_vfadd_vv_f32m1(vsum, vshifted2, vl);

    vfloat32m1_t vshifted3 = __riscv_vslide1down_vf_f32m1(vsum, 0.0f, vl);
    vsum = __riscv_vfadd_vv_f32m1(vsum, vshifted3, vl);

    s = __riscv_vfmv_f_s_f32m1_f32(vsum);
    out[0] = s;
}
#include "harness.h"
