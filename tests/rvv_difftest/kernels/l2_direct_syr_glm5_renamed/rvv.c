#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 0
#define B_IS_MATRIX 0
#define C_IS_MATRIX 1
#define OUT_IS_MATRIX 1
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    for (int i = 0; i < n; i++) {
        float xi = a[i];
        for (int j = 0; j < n; ) {
            vuint32m1_t j_vec = __riscv_vid_v_u32m1(__riscv_vsetvl_e32m1(n - j));
            vbool32_t mask = __riscv_vmsltu_vx_u32m1_b32(j_vec, (unsigned int)(n - j), __riscv_vsetvl_e32m1(n - j));
            size_t vl = __riscv_vcpop_m_b32(mask, __riscv_vsetvl_e32m1(n - j));
            vfloat32m1_t a_vec = __riscv_vle32_v_f32m1_mask(mask, &a[j], vl);
            vfloat32m1_t c_vec = __riscv_vle32_v_f32m1_mask(mask, &c[i * n + j], vl);
            vfloat32m1_t res = __riscv_vfmacc_vf_f32m1_mask(mask, c_vec, 2.0f, a_vec, vl);
            __riscv_vse32_v_f32m1_mask(mask, &out[i * n + j], res, vl);
            j += vl;
        }
    }
}
#include "harness.h"
