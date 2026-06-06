#include <riscv_vector.h>
#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b; (void)c;
    float sum = 0.0f; int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(v, v, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(sq, zero, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(red);
        i += (int)vl;
    }
    out[0] = sqrtf(sum);
}
#include "harness.h"
