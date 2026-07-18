#include <riscv_vector.h>
#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)b; (void)c;
    if (n <= 0){ out[0]=0.0f; return; }
    size_t vlmax = __riscv_vsetvl_e32m1((size_t)-1);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    int i=0;
    while (i<n){
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t x = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t ax = __riscv_vfabs_v_f32m1(x, vl);
        acc = __riscv_vfadd_vv_f32m1_tu(acc, acc, ax, vl);
        i += (int)vl;
    }
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(sum);
}
#include "harness.h"
