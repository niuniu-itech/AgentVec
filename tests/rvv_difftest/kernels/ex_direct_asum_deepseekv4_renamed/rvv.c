#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    float sum = 0.0f;
    int vl;
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t abs_vec = __riscv_vfabs_v_f32m1(vec, vl);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(abs_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(red);
        i += vl;
    }
    out[0] = sum;
}
#include "harness.h"
