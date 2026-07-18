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
    int vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (int j = 0; j < n; j++) {
            int row = i + (j <= i ? 0 : j - i);
            int col = j <= i ? j : i;
            int idx = row * n + col;
            vfloat32m8_t a_val = __riscv_vfmv_v_f_f32m8(a[idx], vl);
            vfloat32m8_t x_val = __riscv_vfmv_v_f_f32m8(b[j], vl);
            acc = __riscv_vfmacc_vv_f32m8(acc, a_val, x_val, vl);
        }
        __riscv_vse32_v_f32m8(out + i, acc, vl);
    }
}
#include "harness.h"
