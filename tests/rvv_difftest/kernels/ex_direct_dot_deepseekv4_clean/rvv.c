#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    int i = 0;
    for (; i + __riscv_vsetvl_e32m1(n - i) <= n; ) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vsum = __riscv_vfredusum_vs_f32m1_f32m1(vsum, vprod, vsum, vl);
        i += vl;
    }
    float s = __riscv_vfmv_f_s_f32m1_f32(vsum);
    out[0] = s;
}
#include "harness.h"
