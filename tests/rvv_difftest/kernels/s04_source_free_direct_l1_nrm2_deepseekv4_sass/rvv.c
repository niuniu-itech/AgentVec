#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float sum = 0.0f;
    int vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_sq = __riscv_vfmul_vv_f32m1(vec_a, vec_a, vl);
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_sq, vec_sum, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    }
    float result = __builtin_sqrtf(sum);
    *out = result;
}
#include "harness.h"
