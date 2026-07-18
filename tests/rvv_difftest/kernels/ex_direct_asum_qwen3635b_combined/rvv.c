#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float sum = 0.0f;
    size_t vl;
    vfloat32m1_t vsum = vmv_v_f32_m1(0.0f);
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    size_t i = 0;
    while (i < (size_t)n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl > 0) {
            vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
            vfloat32m1_t vabs = vfabs_v_f32m1(va, vl);
            vsum = vadd_vv_f32m1(vsum, vabs, vl);
            i += vl;
        }
    }
    
    // Reduce the vector sum to a scalar
    float temp[16];
    __riscv_vse32_v_f32m1(temp, vsum, __riscv_vsetvlmax_e32m1());
    
    float total = 0.0f;
    for (size_t j = 0; j < __riscv_vsetvlmax_e32m1(); j++) {
        total += temp[j];
    }
    
    out[0] = total;
}
#include "harness.h"
