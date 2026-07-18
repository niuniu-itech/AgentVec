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
    size_t vl;
    for (int i = 0; i < n; i++) {
        const float *c_row = c + i * n;
        float *out_row = out + i * n;
        float ai = a[i];
        int j = 0;
        while (j < n) {
            vl = vsetvl_e32m1(n - j);
            vfloat32m1_t vc = vle32_v_f32m1(c_row + j, vl);
            vfloat32m1_t vb = vle32_v_f32m1(b + j, vl);
            vfloat32m1_t vai = vfmv_v_f_f32m1(2.0f * ai, vl);
            vfloat32m1_t vout = vfmacc_vf_f32m1(vc, vai, vb, vl);
            vse32_v_f32m1(out_row + j, vout, vl);
            j += vl;
        }
    }
}
#include "harness.h"
