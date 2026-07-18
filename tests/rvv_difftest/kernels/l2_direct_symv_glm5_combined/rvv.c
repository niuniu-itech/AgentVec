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
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        int j = 0;
        for (; j < i; j += __riscv_vsetvl_e32m1(i - j)) {
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i * n + j], __riscv_vsetvl_e32m1(i - j));
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], __riscv_vsetvl_e32m1(i - j));
            vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, __riscv_vfmul_vv_f32m1(va, vb, __riscv_vsetvl_e32m1(i - j)), __riscv_vsetvl_e32m1(i - j));
        }
        for (; j < n; j += __riscv_vsetvl_e32m1(n - j)) {
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[j * n + i], __riscv_vsetvl_e32m1(n - j));
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], __riscv_vsetvl_e32m1(n - j));
            vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, __riscv_vfmul_vv_f32m1(va, vb, __riscv_vsetvl_e32m1(n - j)), __riscv_vsetvl_e32m1(n - j));
        }
        float temp[32]; 
        int vlmax = __riscv_vsetvl_e32m1(32);
        __riscv_vse32_v_f32m1(temp, vsum, vlmax);
        for (int k = 0; k < vlmax; ++k) sum += temp[k];
        out[i] = sum;
    }
}
#include "harness.h"
