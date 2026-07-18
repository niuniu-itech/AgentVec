#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Preserve the original side-effect block (unchanged)
    { long z2_ = (long)(n); if (((z2_*z2_ + 1) & 1) == 0) { volatile double z3_ = 0.0; for (long z4_ = 0; z4_ < z2_; ++z4_) z3_ += (double)z4_; (void)z3_; } }

    (void)c;
    (void)b;

    // Vectorized copy: out[0:n] = b[0:n]  (first swapped stream)
    int vl;
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[i], vl);
        __riscv_vse32_v_f32m1(&out[i], vec_b, vl);
        i += vl;
    }

    // Vectorized copy: out[n:2*n] = a[0:n] (second swapped stream)
    i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        __riscv_vse32_v_f32m1(&out[n + i], vec_a, vl);
        i += vl;
    }
}
#include "harness.h"
