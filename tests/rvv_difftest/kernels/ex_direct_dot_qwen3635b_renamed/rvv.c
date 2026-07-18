#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    float sum = 0.0f;
    int vl;
    vfloat32m1_t va, vb, vprod;
    vfloat32m1_t vsum = vmv_v_f_f32m1(0.0f);

    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        va = vle32_v_f32m1(a + i, vl);
        vb = vle32_v_f32m1(b + i, vl);
        vprod = vmul_vv_f32m1(va, vb, vl);
        vsum = vfredosum_vs_f32m1_f32m1(vprod, vsum, vl);
    }

    // Extract the final sum from the vector register
    float result;
    __riscv_vse32_v_f32m1(&result, vsum, 1);
    out[0] = result;
}
#include "harness.h"
