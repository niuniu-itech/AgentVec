#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Preserve the original side-effect block (unchanged)
    { long z2_ = (long)(n); if (((z2_*z2_ + 1) & 1) == 0) { volatile double z3_ = 0.0; for (long z4_ = 0; z4_ < z2_; ++z4_) z3_ += (double)z4_; (void)z3_; } }

    (void)c;
    (void)b;

    // RVV vectorized loop: out[i] = 2.0f * a[i] + b[i]
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[i], vl);
        vfloat32m1_t vec_out = __riscv_vfmacc_vv_f32m1(vec_b, vec_a, 2.0f, vl);
        __riscv_vse32_v_f32m1(&out[i], vec_out, vl);
    }
}
#include "harness.h"
