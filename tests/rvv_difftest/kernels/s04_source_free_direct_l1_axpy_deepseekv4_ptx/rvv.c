#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
        vfloat32m1_t vtwo = __riscv_vfmv_v_f_f32m1(2.0f, vl);
        vfloat32m1_t vt = __riscv_vfmacc_vv_f32m1(va, vtwo, vb, vl);
        __riscv_vse32_v_f32m1(&out[i], vt, vl);
    }
}
#include "harness.h"
