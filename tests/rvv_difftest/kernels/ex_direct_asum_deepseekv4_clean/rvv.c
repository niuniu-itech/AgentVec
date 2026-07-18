#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    int i = 0;
    for (int vl = __riscv_vsetvl_e32m1(n); i < n; i += vl, vl = __riscv_vsetvl_e32m1(n - i)) {
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_abs = __riscv_vfabs_v_f32m1(vec_a, vl);
        vec_sum = __riscv_vfadd_vv_f32m1(vec_sum, vec_abs, vl);
    }
    float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    out[0] = sum;
}
#include "harness.h"
