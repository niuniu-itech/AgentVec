#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    
    int i = 0;
    for (; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = vld1_v_f32m1(a + i, vl);
        vfloat32m1_t vb = vld1_v_f32m1(b + i, vl);
        
        // out[0:n] gets b (first swapped stream)
        // out[n:2*n] gets a (second stream)
        vst1_v_f32m1(out + i, vb, vl);
        vst1_v_f32m1(out + n + i, va, vl);
    }
}
#include "harness.h"
