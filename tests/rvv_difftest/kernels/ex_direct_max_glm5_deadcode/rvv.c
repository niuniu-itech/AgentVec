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
    int i = 1;
    int remaining = n - 1;

    if (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t v_s = __riscv_vfmv_v_f_f32m1(s, vl);

        while (remaining > 0) {
            vl = __riscv_vsetvl_e32m1(remaining);
            vfloat32m1_t v_a = __riscv_vle32_v_f32m1(&a[i], vl);
            v_s = __riscv_vfmax_vf_f32m1(v_s, a[i], vl);
            v_s = __riscv_vfmax_vv_f32m1(v_s, v_a, vl);
            i += vl;
            remaining -= vl;
        }

        s = __riscv_vfmv_f_s_f32m1_f32(v_s);
        for (size_t j = 1; j < vl; j++) {
            float tmp = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(v_s, j, vl));
            if (tmp > s) s = tmp;
        }
    }

    out[0] = s;
}
#include "harness.h"
