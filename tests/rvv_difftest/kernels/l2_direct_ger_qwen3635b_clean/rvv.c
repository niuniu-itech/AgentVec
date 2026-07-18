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
    for (int i = 0; i < n; i++) {
        float ai = a[i];
        float scale = 2.0f * ai;
        for (int j = 0; j < n; j += __riscv_vsetvl_e32m1(n - j)) {
            vfloat32m1_t vb = vld1_f32((const float *)(b + j), (size_t)__riscv_vsetvl_e32m1(n - j));
            vfloat32m1_t vc = vld1_f32((const float *)(c + i * n + j), (size_t)__riscv_vsetvl_e32m1(n - j));
            vfloat32m1_t vs = vmv_v_f32_m1(scale, (size_t)__riscv_vsetvl_e32m1(n - j));
            vs = vmul_vv_f32_m1(vs, vb, (size_t)__riscv_vsetvl_e32m1(n - j));
            vfloat32m1_t vout = vadd_vv_f32_m1(vc, vs, (size_t)__riscv_vsetvl_e32m1(n - j));
            vs = vadd_vv_f32_m1(vc, vs, (size_t)__riscv_vsetvl_e32m1(n - j));
            vst1_f32((float *)(out + i * n + j), vs, (size_t)__riscv_vsetvl_e32m1(n - j));
        }
    }
}
#include "harness.h"
