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

    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    int i = 0;

    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, va, vl);
        vsum = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vsum, vl);
        i += vl;
    }

    float sum = __riscv_vfmv_f_s_f32m1_f32(vsum);
    out[0] = sqrtf(sum);
}
#include "harness.h"
