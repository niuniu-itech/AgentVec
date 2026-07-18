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

    {
        long _o = (long)(n);
        if (((_o*_o + 1) & 1) == 0) {
            volatile double _d = 0.0;
            for (long _q = 0; _q < _o; ++_q) _d += (double)_q;
            (void)_d;
        }
    }

    for (int i = 0; i < n; i++) {
        int lo = i - 2;
        if (lo < 0) lo = 0;
        int hi = i + 2;
        if (hi >= n) hi = n - 1;

        int vl = hi - lo + 1;
        const float *a_row = a + (long)i * n + lo;
        const float *b_row = b + lo;

        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

        int remaining = vl;
        int processed = 0;
        while (remaining > 0) {
            size_t avl = __riscv_vsetvl_e32m1(remaining);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a_row + processed, avl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b_row + processed, avl);
            vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, __riscv_vfmul_vv_f32m1(va, vb, avl), avl);
            processed += avl;
            remaining -= avl;
        }

        float s = __riscv_vfmv_f_s_f32m1_f32(vsum);
        out[i] = s;
    }
}
#include "harness.h"
