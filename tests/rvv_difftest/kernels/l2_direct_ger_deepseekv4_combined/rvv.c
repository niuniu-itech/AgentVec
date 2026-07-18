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
    // Strip-mine the outer loop over rows
    int row_start = 0;
    while (row_start < n) {
        int row_vl = __riscv_vsetvl_e32m1(n - row_start);
        // Load a[row_start:row_start+row_vl] (vector x segment)
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a + row_start, row_vl);
        
        // For each row, we need to compute the outer product update for all columns
        // We'll process columns in strips as well
        int col_start = 0;
        while (col_start < n) {
            int col_vl = __riscv_vsetvl_e32m1(n - col_start);
            
            // Load b[col_start:col_start+col_vl] (vector y segment)
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b + col_start, col_vl);
            
            // Compute 2.0f * a[row] * b[col] for each element in the current row strip
            vfloat32m1_t vec_prod = __riscv_vfmul_vf_f32m1(vec_b, 2.0f, col_vl);
            vec_prod = __riscv_vfmul_vv_f32m1(vec_prod, __riscv_vfmv_v_f_f32m1(*(a + row_start), row_vl), col_vl);
            
            // Load c[row_start*n + col_start : ...] (matrix A segment)
            vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(c + row_start * n + col_start, col_vl);
            
            // Compute out = c + 2*x*y^T
            vfloat32m1_t vec_out = __riscv_vfadd_vv_f32m1(vec_c, vec_prod, col_vl);
            
            // Store result
            __riscv_vse32_v_f32m1(out + row_start * n + col_start, vec_out, col_vl);
            
            col_start += col_vl;
        }
        row_start += row_vl;
    }
}
#include "harness.h"
