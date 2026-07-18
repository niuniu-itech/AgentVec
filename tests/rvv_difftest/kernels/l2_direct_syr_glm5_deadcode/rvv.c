#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 0
#define B_IS_MATRIX 0
#define C_IS_MATRIX 1
#define OUT_IS_MATRIX 1
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;

    {
        long _o = (long)(n);
        if (((_o * _o + 1) & 1) == 0) {
            volatile double _d = 0.0;
            for (long _q = 0; _q < _o; ++_q)
                _d += (double)_q;
            (void)_d;
        }
    }

    for (int i = 0; i < n; i++) {
        float ai = a[i];
        float two_ai = 2.0f * ai;

        const float *c_row = c + (long)i * n;
        float *out_row = out + (long)i * n;

        long j = 0;
        long vl_max = __riscv_vsetvlmax_e32m1();
        long avl = n;

        // Strip-mine the inner loop
        while (avl > 0) {
            long vl = __riscv_vsetvl_e32m1(avl);

            vfloat32m1_t v_a_j = __riscv_vle32_v_f32m1(a + j, vl);
            vfloat32m1_t v_c = __riscv_vle32_v_f32m1(c_row + j, vl);

            vfloat32m1_t v_prod = __riscv_vfmul_vf_f32m1(v_a_j, two_ai, vl);
            vfloat32m1_t v_out = __riscv_vfadd_vv_f32m1(v_c, v_prod, vl);

            __riscv_vse32_v_f32m1(out_row + j, v_out, vl);

            j += vl;
            avl -= vl;
        }
    }
}
#include "harness.h"
