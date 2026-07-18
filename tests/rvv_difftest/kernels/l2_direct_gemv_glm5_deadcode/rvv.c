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
        const float *a_row = a + (long)i * n;
        float s = 0.0f;

        long j = 0;
        long avl = n;
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(avl));

        while (avl > 0) {
            long vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a_row + j, vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j, vl);
            vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, __riscv_vfmul_vv_f32m1(va, vb, vl), vl);
            j += vl;
            avl -= vl;
        }

        long avl2 = n;
        while (avl2 > 0) {
            long vl = __riscv_vsetvl_e32m1(avl2);
            vfloat32m1_t vtmp = __riscv_vle32_v_f32m1(&s, vl);
            vtmp = __riscv_vfadd_vv_f32m1(vtmp, vsum, vl);
            __riscv_vse32_v_f32m1(&s, vtmp, vl);
            avl2 -= vl;
        }

        out[i] = s;
    }
}
#include "harness.h"
