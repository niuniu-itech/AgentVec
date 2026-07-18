#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float max_val = -__builtin_inff();
    int vl;
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(c + i, vl);
        // Compute a[i] + b[i] * c[i] (common pattern, though not specified)
        vfloat32m1_t vec_result = __riscv_vfmacc_vv_f32m1(vec_a, vec_b, vec_c, vl);
        // Find max in this vector
        float partial_max = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vec_result, __riscv_vfmv_v_f_f32m1(-__builtin_inff(), vl), vl));
        if (partial_max > max_val) max_val = partial_max;
        i += vl;
    }
    out[0] = max_val;
}
#include "harness.h"
