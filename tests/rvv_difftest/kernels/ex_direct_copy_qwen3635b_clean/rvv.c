#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    int vl;
    vfloat32m1_t va;
    vfloat32m1_t vout;
    
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        va = __riscv_vle_v_f32m1(a + i, vl);
        __riscv_vse_v_f32m1(out + i, va, vl);
    }
}
#include "harness.h"
