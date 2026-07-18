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
#define BANDED_A_RADIUS 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    for (int i = 0; i < n; ) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        
        // Process each element in the current vector group
        for (int j = 0; j < vl; ++j) {
            int idx = i + j;
            int lo = idx - 2;
            if (lo < 0) lo = 0;
            int hi = idx + 2;
            if (hi >= n) hi = n - 1;
            
            vfloat32m1_t vrow = __riscv_vle32_v_f32m1(&a[idx * n + lo], hi - lo + 1);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[lo], hi - lo + 1);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vrow, vb, hi - lo + 1);
            float sum = 0.0f;
            for (int k = 0; k < hi - lo + 1; ++k) {
                sum += vprod[k];
            }
            vsum[j] = sum;
        }
        
        __riscv_vse32_v_f32m1(&out[i], vsum, vl);
        i += vl;
    }
}
#include "harness.h"
