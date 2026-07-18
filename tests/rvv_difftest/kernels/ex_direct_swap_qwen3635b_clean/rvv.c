#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int vl = __riscv_vsetvl_e32m1(n);
    while (vl > 0) {
        vfloat32m1_t vb = __riscv_vle_v_f32m1(b, vl);
        vfloat32m1_t va = __riscv_vle_v_f32m1(a, vl);
        __riscv_vse_v_f32m1(out, vb, vl);
        __riscv_vse_v_f32m1(out + n, va, vl);
        b += vl;
        a += vl;
        out += vl;
        vl = __riscv_vsetvl_e32m1(n - (out - (const float *)out + (const float *)out - out));
        // Recompute remaining elements
        int remaining = n - (int)(out - (const float *)out);
        if (remaining <= 0) break;
        vl = __riscv_vsetvl_e32m1(remaining);
    }
}
#include "harness.h"
