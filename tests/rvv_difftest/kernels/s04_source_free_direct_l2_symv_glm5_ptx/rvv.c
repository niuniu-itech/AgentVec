#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 1
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)
{
    size_t vlmax = vsetvlmax_e32m1();
    for (int i = 0; i < n; i++) {
        vfloat32m1_t v_acc = vfmv_v_f_f32m1(0.0f, vlmax);
        for (int j = 0; j < n; j += vlmax) {
            size_t vl = vsetvl_e32m1(n - j);
            int row = i > j ? i : j;
            int col = i > j ? j : i;
            vfloat32m1_t v_a = vle32_v_f32m1(&a[row * n + col], vl);
            vfloat32m1_t v_x = vle32_v_f32m1(&b[j], vl);
            v_acc = vfmacc_vv_f32m1(v_acc, v_a, v_x, vl);
        }
        out[i] = vfmv_f_s_f32m1_f32(v_acc);
    }
}
#include "harness.h"
