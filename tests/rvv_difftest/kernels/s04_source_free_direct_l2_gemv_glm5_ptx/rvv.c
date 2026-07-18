#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    size_t vl;
    for (size_t i = 0; i < (size_t)n; i++) {
        const float *a_row = a + i * n;
        size_t j = 0;
        vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, 1);
        for (; j < (size_t)n; j += vl) {
            vl = vsetvl_e32m1((size_t)n - j);
            vfloat32m1_t va = vle32_v_f32m1(a_row + j, vl);
            vfloat32m1_t vb = vle32_v_f32m1(b + j, vl);
            acc = vfmacc_vf_f32m1(acc, 1.0f, va, vl);
            acc = vfmacc_vv_f32m1(acc, vb, vl);
        }
        vl = vsetvl_e32m1(1);
        vfloat32m1_t vout = vfmv_v_f_f32m1(0.0f, vl);
        vout = vfredsum_vs_f32m1_f32m1(vout, acc, vout, vl);
        float sum = vfmv_f_s_f32m1_f32(vout);
        out[i] = sum + c[i];
    }
}
#include "harness.h"
