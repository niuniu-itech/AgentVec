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

    // RVV vector reduction: dot product of a and b
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    size_t vl;
    int i = 0;
    for (; (vl = __riscv_vsetvl_e32m1(n - i)) > 0; i += vl) {
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vec_sum = __riscv_vfadd_vv_f32m1(vec_sum, vprod, vl);
    }
    // Reduce vector to scalar
    float result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vec_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl));
    out[0] = result;
}
#include "harness.h"
