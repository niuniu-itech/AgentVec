#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int i = 0;
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    for (i = 0; i < n; i++) {
        int j = 0;
        vfloat32m1_t vsum = vzero;
        
        while (j < n) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t vb = vle32_v_f32m1(b + j, vl);
            vfloat32m1_t va = vle32_v_f32m1(a + i * n + j, vl);
            vsum = vfmacc_vv_f32m1(vsum, va, vb, vl);
            j += vl;
        }
        
        float s = 0.0f;
        int vl = __riscv_vsetvl_e32m1(1);
        vse32_v_f32m1(&s, vsum, vl);
        out[i] = s;
    }
}
#include "harness.h"
