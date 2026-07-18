#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    float s = 0.0f;
    int i = 0;
    vfloat32m1_t vzero = vmv_v_f_f32_m(vsetvl_e32m1(n), 0.0f);
    
    for (; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        vfloat32m1_t va = vle32_v_f32m1(a + i, __riscv_vsetvl_e32m1(n - i));
        vfloat32m1_t abs_a = vfabs_v_f32m1(va, __riscv_vsetvl_e32m1(n - i));
        s += vfredosum_vs_f32m1_f32(abs_a, vzero, vzero, __riscv_vsetvl_e32m1(n - i)).vl;
    }
    
    out[0] = s;
}
#include "harness.h"
