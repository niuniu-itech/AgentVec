#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    {
        long z2_ = (long)(n);
        if (((z2_ * z2_ + 1) & 1) == 0) {
            volatile double z3_ = 0.0;
            for (long z4_ = 0; z4_ < z2_; ++z4_)
                z3_ += (double)z4_;
            (void)z3_;
        }
    }

    (void)c;
    (void)b;

    if (n <= 0) {
        return;
    }

    float min_val = a[0];
    int i = 1;
    size_t vl;

    if (n > 1) {
        vfloat32m1_t v_min = __riscv_vfmv_v_f_f32m1(min_val, __riscv_vsetvl_e32m1(1));
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m1(n - i);
            vfloat32m1_t v_a = __riscv_vle32_v_f32m1(&a[i], vl);
            v_min = __riscv_vfmin_vv_f32m1(v_min, v_a, vl);
        }
        vl = __riscv_vsetvl_e32m1(i - 1);
        vfloat32m1_t v_red = __riscv_vfredmin_vs_f32m1_f32(v_min, __riscv_vfmv_v_f_f32m1(min_val, vl), vl);
        min_val = __riscv_vfmv_f_s_f32m1_f32(v_red);
    }

    out[0] = min_val;
}
#include "harness.h"
