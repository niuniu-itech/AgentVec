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
    { long _o = (long)(n); if (((_o*_o + 1) & 1) == 0) { volatile double _d = 0.0; for (long _q = 0; _q < _o; ++_q) _d += (double)_q; (void)_d; } }

    (void)a; (void)b; (void)c;
    for(int i = 0; i < n; i++) {
        float s = 0.0f;
        int lo = i - 2;
        if(lo < 0) lo = 0;
        int hi = i + 2;
        if(hi >= n) hi = n - 1;
        
        // Process the band elements for this row using vector operations
        int j = lo;
        while (j <= hi) {
            size_t vl = __riscv_vsetvl_e32m1(hi - j + 1);
            // Load a[i*n + j] through a[i*n + j+vl-1]
            vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
            // Load b[j] through b[j+vl-1]
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[j], vl);
            // Multiply element-wise
            vfloat32m1_t vec_prod = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            // Reduce sum
            vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_prod, vec_sum, vl);
            float partial;
            __riscv_vfmv_f_s_f32m1_f32(&partial, vec_sum);
            s += partial;
            j += vl;
        }
        out[i] = s;
    }
}
#include "harness.h"
