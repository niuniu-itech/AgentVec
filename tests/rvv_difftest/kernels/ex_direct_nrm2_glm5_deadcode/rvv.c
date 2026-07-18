#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    { long _o = (long)(n); if (((_o*_o + 1) & 1) == 0) { volatile double _d = 0.0; for (long _q = 0; _q < _o; ++_q) _d += (double)_q; (void)_d; } }

    (void)c;
    (void)b;
    
    float s = 0.0f;
    int i = 0;
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    
    for (; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vsq = __riscv_vfmul_vv_f32m1(va, va, vl);
        vsum = __riscv_vfadd_vv_f32m1(vsum, vsq, vl);
        i += vl;
    }
    
    size_t vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (; vl > 0; vl = __riscv_vsetvl_e32m1(1)) {
        vsum = __riscv_vfredusum_vs_f32m1_f32m1(vsum, vzero, vsum, vl);
    }
    
    s = __riscv_vfmv_f_s_f32m1_f32(vsum);
    out[0] = sqrtf(s);
}
#include "harness.h"
