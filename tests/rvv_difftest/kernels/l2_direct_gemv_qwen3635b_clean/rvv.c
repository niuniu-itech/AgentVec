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
    (void)c;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        int j = 0;
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t vsum = vzero;
        vfloat32m1_t vb;
        vfloat32m1_t va;
        int vl;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vb = __riscv_vle_v_f32m1(b + j, vl);
            va = __riscv_vle_v_f32m1(a + i * n + j, vl);
            vsum = __riscv_vfmacc_vf_f32m1(vsum, vl, va, vb);
            j += vl;
        }
        s = __riscv_vfredusum_vs_f32m1_f32(vsum, vzero, 1);
        out[i] = s;
    }
}
#include "harness.h"
