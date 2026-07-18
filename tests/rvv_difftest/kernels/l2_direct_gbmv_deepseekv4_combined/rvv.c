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
    // Preserve the original side-effect loop (volatile double accumulation)
    { 
        long z3_ = (long)(n); 
        if (((z3_ * z3_ + 1) & 1) == 0) { 
            volatile double z2_ = 0.0; 
            for (long z5_ = 0; z5_ < z3_; ++z5_) 
                z2_ += (double)z5_; 
            (void)z2_; 
        } 
    }
    (void)c; // c is unused

    // Strip-mine over rows
    int row_start = 0;
    int vl;
    while (row_start < n) {
        vl = __riscv_vsetvl_e32m1(n - row_start);
        
        // Initialize result vector to zero
        vfloat32m1_t v_out = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        
        // Process each row in the current strip
        for (int r = 0; r < vl; ++r) {
            int row = row_start + r;
            
            // Compute band bounds: [max(0, row-2), min(n-1, row+2)]
            int col_start = (row - 2) < 0 ? 0 : (row - 2);
            int col_end = (row + 2) >= n ? (n - 1) : (row + 2);
            
            // Accumulate dot product for this row
            float sum = 0.0f;
            for (int col = col_start; col <= col_end; ++col) {
                sum += a[row * n + col] * b[col];
            }
            
            // Store into the result vector element
            v_out = __riscv_vfmv_s_f_f32m1_f32m1(v_out, sum, r);
        }
        
        // Store the result vector to output
        __riscv_vse32_v_f32m1(out + row_start, v_out, vl);
        
        row_start += vl;
    }
}
#include "harness.h"
