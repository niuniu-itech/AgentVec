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

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        int j = 0;
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));
        for (; j + 3 < n; j += 4) {
            int j0 = j, j1 = j + 1, j2 = j + 2, j3 = j + 3;
            float a0 = (j0 <= i) ? a[i * n + j0] : a[j0 * n + i];
            float a1 = (j1 <= i) ? a[i * n + j1] : a[j1 * n + i];
            float a2 = (j2 <= i) ? a[i * n + j2] : a[j2 * n + i];
            float a3 = (j3 <= i) ? a[i * n + j3] : a[j3 * n + i];
            float vals[4] = {a0, a1, a2, a3};
            vfloat32m1_t vvals = __riscv_vle32_v_f32m1(vals, __riscv_vsetvl_e32m1(4));
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], __riscv_vsetvl_e32m1(4));
            vsum = __riscv_vfmacc_vf_f32m1(vsum, s, vvals, __riscv_vsetvl_e32m1(4));
            vsum = __riscv_vfmacc_vv_f32m1(vsum, vb, vvals, __riscv_vsetvl_e32m1(4));
        }
        for (; j < n; j++) {
            float v = (j <= i) ? a[i * n + j] : a[j * n + i];
            s += v * b[j];
        }
        vfloat32m1_t vred = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        vred = __riscv_vfredusum_vs_f32m1_f32m1(vred, vsum, vred, __riscv_vsetvl_e32m1(n));
        out[i] = s + __riscv_vfmv_f_s_f32m1_f32(vred);
    }
}
#include "harness.h"
