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
        vint32m1_t v_i = __riscv_vid_v_i32m1(__riscv_vsetvl_e32m1(n - i));
        vbool32_t v_mask = __riscv_vmslt_vx_i32m1_b32(v_i, n - i, __riscv_vsetvl_e32m1(n - i));
        size_t vl = __riscv_vcpop_m_b32(v_mask, __riscv_vsetvl_e32m1(n - i));

        vfloat32m1_t v_a = __riscv_vle32_v_f32m1_m(v_mask, a + i, vl);
        vfloat32m1_t v_res = __riscv_vfmul_vf_f32m1_m(v_mask, v_a, 2.0f, vl);
        __riscv_vse32_v_f32m1_m(v_mask, out + i, v_res, vl);

        i += vl;
    }
}
#include "harness.h"
