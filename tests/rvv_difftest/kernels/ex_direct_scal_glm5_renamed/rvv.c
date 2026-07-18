#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    int i = 0;
    while (i < n) {
        vint32m1_t vvl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vvl);
        vfloat32m1_t vres = __riscv_vfmul_vf_f32m1(va, 2.0f, vvl);
        __riscv_vse32_v_f32m1(out + i, vres, vvl);
        i += __riscv_vmv_x_s_i32m1_i32(vvl);
    }
}
#include "harness.h"
