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

    float s = 0.0f;
    int i = 0;
    vfloat32m1_t va, vsq;
    vfloat32m1_t vsum = vmv_v_f32m1(0.0f);

    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;

        va = vle32_v_f32m1(a + i, vl);
        vsq = vmul_vv_f32m1(va, va, vl);
        vsum = vfredosum_vs_f32m1_f32m1(vsum, vsq, vsum, vl);

        i += (int)vl;
    }

    // Extract the final sum from the vector register
    // vfredosum accumulates into the first lane of the destination vector
    float result = vget_v_f32m1_x(vsum);
    out[0] = sqrtf(result);
}
#include "harness.h"
