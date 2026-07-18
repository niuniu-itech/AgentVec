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
    for (int i = 0; i < n; i++) {
        int lo = i - 2;
        if (lo < 0) lo = 0;
        int hi = i + 2;
        if (hi >= n) hi = n - 1;
        int len = hi - lo + 1;
        
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        int j = lo;
        while (j <= hi) {
            int vl = __riscv_vsetvl_e32m1(hi - j + 1);
            vfloat32m1_t a_vals = __riscv_vlse32_v_f32m1(&a[i * n + j], sizeof(float) * n, vl);
            vfloat32m1_t b_vals = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(a_vals, b_vals, vl);
            acc = __riscv_vfadd_vv_f32m1_tu(acc, acc, prod, vl);
            j += vl;
        }
        vfloat32m1_t result = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        result = __riscv_vfredusum_vs_f32m1_f32m1(acc, result, __riscv_vsetvl_e32m1(1));
        out[i] = __riscv_vfmv_f_s_f32m1_f32(result);
    }
}
#include "harness.h"
