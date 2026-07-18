#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    float s = 0.0f;
    int i = 0;
    while (i < n) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t abs_a = __riscv_vfabs_v_f32m1(vec_a, vl);
        vfloat32m1_t sum_vec = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        sum_vec = __riscv_vfadd_vv_f32m1(sum_vec, abs_a, vl);
        s += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(sum_vec, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl));
        i += vl;
    }
    out[0] = s;
}
#include "harness.h"
