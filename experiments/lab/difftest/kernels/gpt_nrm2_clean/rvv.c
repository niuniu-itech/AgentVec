#include <riscv_vector.h>
#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)b; (void)c;
    size_t vl = __riscv_vsetvl_e32m1((size_t)n);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (int i = 0; i < n; ) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(va, va, vl);
        acc = __riscv_vfadd_vv_f32m1(acc, sq, vl);
        i += (int)vl;
    }
    vl = __riscv_vsetvl_e32m1((size_t)n);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t sumv = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl);
    float sum = __riscv_vfmv_f_s_f32m1_f32(sumv);
    out[0] = sqrtf(sum);
}
#include "harness.h"
