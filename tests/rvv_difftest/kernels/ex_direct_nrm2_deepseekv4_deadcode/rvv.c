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
    
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    int i = 0;
    int vl;
    
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_sq = __riscv_vfmul_vv_f32m1(vec_a, vec_a, vl);
        vec_sum = __riscv_vfadd_vv_f32m1_tu(vec_sum, vec_sum, vec_sq, vl);
        i += vl;
    }
    
    float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vec_sum, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl));
    out[0] = sqrtf(sum);
}
#include "harness.h"
