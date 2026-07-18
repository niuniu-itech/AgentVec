#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Preserve the original side-effect loop (unchanged semantics)
    {
        long z2_ = (long)n;
        if (((z2_ * z2_ + 1) & 1) == 0) {
            volatile double z3_ = 0.0;
            for (long z4_ = 0; z4_ < z2_; ++z4_)
                z3_ += (double)z4_;
            (void)z3_;
        }
    }

    (void)c;
    (void)b;

    // Vectorized dot product: sum of a[i] * a[i]
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    size_t vl;
    int i = 0;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_prod = __riscv_vfmul_vv_f32m1(vec_a, vec_a, vl);
        vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_prod, vec_sum, vl);
    }
    // Extract scalar sum from the reduction result
    float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    out[0] = sqrtf(sum);
}
#include "harness.h"
