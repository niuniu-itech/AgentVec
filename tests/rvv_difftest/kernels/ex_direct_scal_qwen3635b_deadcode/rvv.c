#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    int i = 0;
    vfloat32m1_t v2 = vfmv_v_f_f32m1(2.0f);
    
    for (i = 0; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vout = vfmul_vv_f32m1(va, v2, vl);
        vse32_v_f32m1(out + i, vout, vl);
    }
}
#include "harness.h"
