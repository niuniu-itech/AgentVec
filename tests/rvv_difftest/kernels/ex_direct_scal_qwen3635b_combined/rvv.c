#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t va, vout;
    while (vl > 0) {
        va = vle32_v_f32m1(a, vl);
        vout = vmul_vf_f32m1(va, 2.0f, vl);
        vse32_v_f32m1(out, vout, vl);
        a += vl;
        out += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
}
#include "harness.h"
