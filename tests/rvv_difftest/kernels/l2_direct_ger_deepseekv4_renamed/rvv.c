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
    // a = x, b = y, c = A (row-major), out = A + 2*x*y^T
    for (int i = 0; i < n; ++i) {
        float xi = a[i];
        float two_xi = 2.0f * xi;
        
        // Process each row i: out[i*n + j] = c[i*n + j] + 2*xi*y[j]
        const float *row_c = c + i * n;
        float *row_out = out + i * n;
        
        int j = 0;
        while (j < n) {
            size_t vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load y[j:j+vl]
            vfloat32m1_t y_vec = __riscv_vle32_v_f32m1(b + j, vl);
            
            // Load A[i][j:j+vl]
            vfloat32m1_t a_row_vec = __riscv_vle32_v_f32m1(row_c + j, vl);
            
            // Compute 2*xi * y[j:j+vl]
            vfloat32m1_t scaled_y = __riscv_vfmul_vf_f32m1(y_vec, two_xi, vl);
            
            // Compute A[i][j:j+vl] + 2*xi*y[j:j+vl]
            vfloat32m1_t result = __riscv_vfadd_vv_f32m1(a_row_vec, scaled_y, vl);
            
            // Store result
            __riscv_vse32_v_f32m1(row_out + j, result, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
