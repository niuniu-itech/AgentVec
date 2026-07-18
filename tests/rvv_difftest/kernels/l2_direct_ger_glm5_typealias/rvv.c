#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 0
#define B_IS_MATRIX 0
#define C_IS_MATRIX 1
#define OUT_IS_MATRIX 1
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float a_i = a[i];
        vfloat32m1_t va = __riscv_vfmv_v_f_f32m1(2.0f * a_i, __riscv_vsetvl_e32m1(1));
        
        for (int j = 0; j < n; ) {
            size_t vl = __riscv_vsetvl_e32m1(n - j);
            
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vc = __riscv_vle32_v_f32m1(&c[i * n + j], vl);
            
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
            vfloat32m1_t vres  = __riscv_vfadd_vv_f32m1(vc, vprod, vl);
            
            __riscv_vse32_v_f32m1(&out[i * n + j], vres, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
