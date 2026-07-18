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
    int i = 0;
    size_t vl;

    // Strip-mine the array using vsetvl_e32m1
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);

        // Load vector of floats
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);

        // Multiply each element by itself (a[i] * a[i])
        vfloat32m1_t vsq = __riscv_vfmul_vv_f32m1(va, va, vl);

        // Horizontal sum of the squared elements
        sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vsq, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl));

        i += vl;
    }

    // Write the final scalar result
    out[0] = sqrtf(sum);
}
#include "harness.h"
