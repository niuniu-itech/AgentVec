#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    { long _o = (long)(n); if (((_o*_o + 1) & 1) == 0) { volatile double _d = 0.0; for (long _q = 0; _q < _o; ++_q) _d += (double)_q; (void)_d; } }

    (void)c;
    (void)b;
    
    int vl;
    int i = 0;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b + i, vl);
        __riscv_vse32_v_f32m1(out + i, vec_b, vl);
        
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a + i, vl);
        __riscv_vse32_v_f32m1(out + n + i, vec_a, vl);
    }
}
#include "harness.h"
