#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int vl = __riscv_vsetvl_e32m1(n);
    while (vl > 0) {
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = vfmv_v_f_f32m1(2.0f);
        vfloat32m1_t vres = vfmul_vv_f32m1(va, vc, vl);
        vfloat32m1_t vout = vfadd_vv_f32m1(vres, vb, vl);
        vse32_v_f32m1(out, vout, vl);
        a += vl;
        b += vl;
        out += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
}
#include "harness.h"
