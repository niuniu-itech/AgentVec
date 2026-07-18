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
#define BANDED_A_RADIUS 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        int j_start = i - 2;
        if (j_start < 0) j_start = 0;
        int j_end = i + 2;
        if (j_end >= n) j_end = n - 1;
        int len = j_end - j_start + 1;

        float sum = 0.0f;
        int j = j_start;
        int remaining = len;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m1(remaining);
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_prod = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            sum = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m1_f32m1(vec_prod, __riscv_vfmv_s_f_f32m1(sum, vl), vl),
                vl
            );
            j += vl;
            remaining -= vl;
        }
        out[i] = sum;
    }
}
#include "harness.h"
