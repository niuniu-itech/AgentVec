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
    // Strip-mine the outer loop over rows
    for (int i = 0; i < n; i++) {
        // Initialize vector accumulator to zero
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        
        // Strip-mine the inner loop over columns
        int j = 0;
        while (j < n) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load a[i*n + j : i*n + j + vl] and b[j : j + vl]
            vfloat32m1_t a_vec = __riscv_vlse32_v_f32m1(&a[i * n + j], sizeof(float) * n, vl);
            vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(&b[j], vl);
            
            // Multiply and accumulate
            acc = __riscv_vfmacc_vv_f32m1(acc, a_vec, b_vec, vl);
            
            j += vl;
        }
        
        // Reduce the vector accumulator to a scalar and store
        vfloat32m1_t sum_vec = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        sum_vec = __riscv_vfredusum_vs_f32m1_f32m1(acc, sum_vec, __riscv_vsetvl_e32m1(n));
        float result = __riscv_vfmv_f_s_f32m1_f32(sum_vec);
        out[i] = result;
    }
}
#include "harness.h"
