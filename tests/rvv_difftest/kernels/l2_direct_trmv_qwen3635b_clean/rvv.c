#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 1
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int i = 0;
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    for (i = 0; i < n; i++) {
        // For row i, we need to compute sum_{j=0}^{i} a[i*n + j] * b[j]
        // The number of elements to process is i+1
        int count = i + 1;
        int remaining = count;
        const float *a_row = &a[i * n];
        float s = 0.0f;
        
        while (remaining > 0) {
            vltype_t vl = __riscv_vsetvl_e32m1(remaining);
            vfloat32m1_t va = vle32_v_f32m1(a_row, vl);
            vfloat32m1_t vb = vle32_v_f32m1(b, vl);
            vfloat32m1_t vprod = vmul_vv_f32m1(va, vb, vl);
            s += vfredosum_vs_f32m1_f32(vprod, vzero, vl)[0];
            a_row += vl;
            b += vl;
            remaining -= vl;
        }
        out[i] = s;
    }
}
#include "harness.h"
