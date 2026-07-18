#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float acc = -__builtin_inff();
    int vl;
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[i], vl);
        vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i], vl);
        // Compute something (example: a + b * c)
        vfloat32m1_t vec_result = __riscv_vfmacc_vv_f32m1(vec_a, vec_b, vec_c, vl);
        __riscv_vse32_v_f32m1(&out[i], vec_result, vl);
        i += vl;
    }
}
#include "harness.h"
