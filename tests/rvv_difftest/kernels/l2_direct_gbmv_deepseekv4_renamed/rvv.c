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
        int start = i - 2;
        if (start < 0) start = 0;
        int end = i + 2;
        if (end >= n) end = n - 1;
        
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        int j = start;
        int vl;
        for (; j <= end; j += vl) {
            vl = __riscv_vsetvl_e32m1(end - j + 1);
            vfloat32m1_t a_vals = __riscv_vlse32_v_f32m1(&a[i * n + j], sizeof(float) * n, vl);
            vfloat32m1_t b_vals = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(a_vals, b_vals, vl);
            sum = __riscv_vfredusum_vs_f32m1_f32m1(prod, sum, vl);
        }
        out[i] = __riscv_vfmv_f_s_f32m1_f32(sum);
    }
}
#include "harness.h"
