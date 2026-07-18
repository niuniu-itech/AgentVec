#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    if (n <= 0) {
        out[0] = 0.0f;
        return;
    }

    float max_val = a[0];
    int vl = __riscv_vsetvl_e32m1(n);
    int i = 0;

    while (i < n) {
        if (vl > 0) {
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
            vfloat32m1_t vmax = __riscv_vmax_vv_f32m1(va, va, vl);
            float local_max = __riscv_vreduce_max_f32_m1(vmax, vl);
            if (local_max > max_val) {
                max_val = local_max;
            }
        }
        i += vl;
        vl = __riscv_vsetvl_e32m1(n - i);
    }

    out[0] = max_val;
}
#include "harness.h"
