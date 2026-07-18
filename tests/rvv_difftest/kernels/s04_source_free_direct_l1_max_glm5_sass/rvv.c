#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(-__builtin_inff(), __riscv_vsetvl_e32m1(1));
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfredmax_vs_f32m1_f32m1(va, acc, vl);
        i += vl;
    }
    float max_val = __riscv_vfmv_f_s_f32m1_f32(acc);
    *out = max_val;
}
#include "harness.h"
