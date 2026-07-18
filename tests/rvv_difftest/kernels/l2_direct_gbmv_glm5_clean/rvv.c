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
        int lo = i - 2;
        if (lo < 0) lo = 0;
        int hi = i + 2;
        if (hi >= n) hi = n - 1;
        int len = hi - lo + 1;

        float s = 0.0f;
        int j = lo;
        while (j <= hi) {
            size_t vl = __riscv_vsetvl_e32m1(len);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
            s = __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(s, vl), vl));
            j += vl;
            len -= vl;
        }
        out[i] = s;
    }
}
#include "harness.h"
