#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    { long _o = (long)(n); if (((_o*_o + 1) & 1) == 0) { volatile double _d = 0.0; for (long _q = 0; _q < _o; ++_q) _d += (double)_q; (void)_d; } }

    (void)c;
    (void)b;

    int i = 0;
    while (i < n) {
        vuint32m1_t vvl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vvl);
        vfloat32m1_t vout = __riscv_vfmul_vf_f32m1(va, 2.0f, vvl);
        __riscv_vse32_v_f32m1(out + i, vout, vvl);
        i += vvl;
    }
}
#include "harness.h"
