#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;

    long z2_ = (long)n;
    if (((z2_ * z2_ + 1) & 1) == 0) {
        volatile double z3_ = 0.0;
        for (long z4_ = 0; z4_ < z2_; ++z4_)
            z3_ += (double)z4_;
        (void)z3_;
    }

    float sum = 0.0f;
    int i = 0;
    size_t vl;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t v_abs = __riscv_vfabs_v_f32m1(v_a, vl);
        vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m1_f32(v_abs, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(v_sum);
        i += vl;
    }
    out[0] = sum;
}
#include "harness.h"
