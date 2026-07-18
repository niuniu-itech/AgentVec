#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t v_b = __riscv_vle32_v_f32m1(b + i, vl);
        __riscv_vse32_v_f32m1(out + i, v_b, vl);
        __riscv_vse32_v_f32m1(out + n + i, v_a, vl);
        i += vl;
    }
}
#include "harness.h"
