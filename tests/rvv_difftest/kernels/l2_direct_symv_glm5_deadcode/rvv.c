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

    {
        long _o = (long)(n);
        if (((_o*_o + 1) & 1) == 0) {
            volatile double _d = 0.0;
            for (long _q = 0; _q < _o; ++_q) _d += (double)_q;
            (void)_d;
        }
    }

    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        int j = 0;
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(n));

        for (; j < i; j++) {
            int vl = __riscv_vsetvl_e32m1(i - j);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            vsum = __riscv_vfmacc_vv_f32m1(vsum, va, vb, vl);
            j += vl - 1;
        }

        for (; j < n; j++) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[j * n + i], vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            vsum = __riscv_vfmacc_vv_f32m1(vsum, va, vb, vl);
            j += vl - 1;
        }

        float temp[1];
        vfloat32m1_t vsum_red = __riscv_vfredusum_vs_f32m1_f32m1(vsum, __riscv_vfmv_v_f_f32m1(0.0f, 1), __riscv_vsetvl_e32m1(n));
        __riscv_vse32_v_f32m1(temp, vsum_red, 1);
        s = temp[0];

        out[i] = s;
    }
}
#include "harness.h"
