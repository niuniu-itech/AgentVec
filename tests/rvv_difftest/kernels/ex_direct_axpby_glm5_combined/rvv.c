#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int i = 0;
    while (i < n) {
        vint32m1_t v_i = __riscv_vid_v_i32m1(__riscv_vsetvl_e32m1(n - i));
        vbool32_t vm = __riscv_vmslt_vv_i32m1_b32(v_i, __riscv_vmv_v_x_i32m1(n - i, __riscv_vsetvl_e32m1(n - i)), __riscv_vsetvl_e32m1(n - i));
        size_t vl = __riscv_vcpop_m_b32(vm, __riscv_vsetvl_e32m1(n - i));
        if (vl == 0) break;

        vfloat32m1_t va = __riscv_vle32_v_f32m1_mask(vm, a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1_mask(vm, b + i, vl);
        vfloat32m1_t vout = __riscv_vfmacc_vf_f32m1_mask(vm, __riscv_vfmul_vf_f32m1_mask(vm, va, 2.0f, vl), 3.0f, vb, vl);
        __riscv_vse32_v_f32m1_mask(vm, out + i, vout, vl);

        i += vl;
    }
}
#include "harness.h"
