#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 1
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Strip-mine the outer loop over rows
    int row = 0;
    while (row < n) {
        int vl = __riscv_vsetvl_e32m1(n - row);
        
        // Initialize accumulator vector for this strip of rows
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        
        // Process columns for this strip of rows
        for (int col = 0; col < n; ++col) {
            // Gather the appropriate element from the symmetric matrix
            // For each row in the strip, we need a[i*n + col] if col <= row, else a[col*n + row]
            // We'll compute the index for each row in the strip
            vfloat32m1_t a_vals;
            
            // Since we need different indices per row, we'll use a scalar loop for the inner
            // product to maintain correctness with the source semantics
            // Actually, we can vectorize the inner product by loading b[col] and multiplying
            // with the correct a element for each row
            
            // For each row in the strip, compute the index into a
            // We'll use a gather approach: for each row r in [row, row+vl-1],
            // the index is (col <= r) ? r*n + col : col*n + r
            // This requires per-lane index computation
            
            // Create vector of row indices
            vuint32m1_t row_indices = __riscv_vid_v_u32m1(vl);
            vuint32m1_t row_offsets = __riscv_vadd_vx_u32m1(row_indices, row, vl);
            
            // Compute column index vector (all same column)
            vuint32m1_t col_vec = __riscv_vmv_v_x_u32m1(col, vl);
            
            // Compare: col <= row?
            vbool32_t mask = __riscv_vmseq_vv_u32m1_b32(col_vec, row_offsets, vl); // Actually need <=
            // For <=, we use vmsle
            mask = __riscv_vmsleu_vv_u32m1_b32(col_vec, row_offsets, vl);
            
            // Compute both possible indices
            vuint32m1_t idx1 = __riscv_vmul_vx_u32m1(row_offsets, n, vl);
            idx1 = __riscv_vadd_vv_u32m1(idx1, col_vec, vl);
            
            vuint32m1_t idx2 = __riscv_vmul_vx_u32m1(col_vec, n, vl);
            idx2 = __riscv_vadd_vv_u32m1(idx2, row_offsets, vl);
            
            // Select based on mask
            vuint32m1_t indices = __riscv_vmerge_vvm_u32m1(idx1, idx2, mask, vl);
            
            // Gather from a
            a_vals = __riscv_vluxei32_v_f32m1(a, indices, vl);
            
            // Load b[col] (scalar broadcast)
            vfloat32m1_t b_val = __riscv_vfmv_v_f_f32m1(b[col], vl);
            
            // Multiply and accumulate
            acc = __riscv_vfmacc_vv_f32m1(acc, a_vals, b_val, vl);
        }
        
        // Store results
        __riscv_vse32_v_f32m1(out + row, acc, vl);
        
        row += vl;
    }
}
#include "harness.h"
