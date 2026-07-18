#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    while (vl > 0) {
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b, vl);
        vfloat32m1_t va_mul = vmul_vf_f32m1(va, 2.0f, vl);
        vfloat32m1_t vb_mul = vmul_vf_f32m1(vb, 3.0f, vl);
        vfloat32m1_t res = vadd_vv_f32m1(va_mul, vb_mul, vl);
        vse32_v_f32m1(out, res, vl);
        a += vl;
        b += vl;
        out += vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
}
#include "harness.h"
