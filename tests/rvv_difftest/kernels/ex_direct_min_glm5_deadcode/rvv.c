#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;

    if (n <= 0) {
        return;
    }

    float s = a[0];
    size_t vl = __riscv_vsetvl_e32m1(n - 1);
    vfloat32m1_t vmin = __riscv_vfmv_v_f_f32m1(s, vl);

    int i = 1;
    for (; i < n;) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vmin = __riscv_vfmin_vv_f32m1(vmin, va, vl);
        i += vl;
    }

    s = __riscv_vfmv_f_s_f32m1_f32(vmin);
    out[0] = s;
}
#include "harness.h"
