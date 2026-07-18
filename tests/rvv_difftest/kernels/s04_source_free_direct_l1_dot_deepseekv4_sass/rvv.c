#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float acc = 0.0f;
    int vl;
    int i = 0;
    
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
        acc = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(acc, vl), vl));
        i += vl;
    }
    
    *out = acc;
}
#include "harness.h"
