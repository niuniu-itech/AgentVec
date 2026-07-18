#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Preserve the original side-effect computation (unchanged)
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

    // Vectorized reduction: sum of absolute values of a[0..n-1]
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    vfloat32m1_t acc = vec_zero;
    int vl;
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_abs = __riscv_vfabs_v_f32m1(vec_a, vl);
        acc = __riscv_vfadd_vv_f32m1(acc, vec_abs, vl);
        i += vl;
    }
    // Reduce vector to scalar
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(acc, vec_zero, vl);
    float result = __riscv_vfmv_f_s_f32m1_f32(sum);
    out[0] = result;
}
#include "harness.h"
