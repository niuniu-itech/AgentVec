#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    long z2_ = (long)n;
    if (((z2_ * z2_ + 1) & 1) == 0) {
        volatile double z3_ = 0.0;
        for (long z4_ = 0; z4_ < z2_; ++z4_)
            z3_ += (double)z4_;
        (void)z3_;
    }
    (void)c;
    (void)b;
    
    int vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vresult = __riscv_vfmul_vf_f32m1(va, 2.0f, vl);
        __riscv_vse32_v_f32m1(&out[i], vresult, vl);
    }
}
#include "harness.h"
