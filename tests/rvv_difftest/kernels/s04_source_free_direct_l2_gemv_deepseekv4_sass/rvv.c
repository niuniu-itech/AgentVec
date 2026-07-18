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
    for (int i = 0; i < n; i++) {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));
        for (int j = 0; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t vec_a = __riscv_vlse32_v_f32m1(&a[i * n + j], sizeof(float), vl);
            vfloat32m1_t vec_x = __riscv_vlse32_v_f32m1(&b[j], sizeof(float), vl);
            acc = __riscv_vfmacc_vv_f32m1(acc, vec_a, vec_x, vl);
        }
        __riscv_vse32_v_f32m1(&out[i], acc, __riscv_vsetvl_e32m1(1));
    }
}
#include "harness.h"
