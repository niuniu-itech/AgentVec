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
    
    // Initialize max with first element
    float max_val = a[0];
    
    // Vector reduction to find maximum
    int i = 1;
    size_t vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        // Broadcast current max to vector
        vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(max_val, vl);
        // Compare and select larger values
        vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(vec_max, vec_a, vl);
        vec_max = __riscv_vfmerge_vvm_f32m1(vec_max, vec_a, mask, vl);
        // Reduce vector to scalar max
        max_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vec_max, __riscv_vfmv_s_f_f32m1(max_val), vl));
    }
    
    out[0] = max_val;
}
#include "harness.h"
