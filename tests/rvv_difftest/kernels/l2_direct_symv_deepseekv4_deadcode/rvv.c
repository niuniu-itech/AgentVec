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
    // Strip-mine over rows i = 0..n-1
    for (int i = 0; i < n; ) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        // Load b[j] for j = i..i+vl-1 (the part of b needed for this chunk)
        vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b + i, vl);
        
        // We'll accumulate partial sums for each row in this chunk
        // For each row i+k (k=0..vl-1), we need to compute dot product with a's row/column
        // We'll process one row at a time within the chunk
        for (int k = 0; k < vl; ++k) {
            int row = i + k;
            float sum = 0.0f;
            
            // Process columns j = 0..row (lower triangle part)
            int j = 0;
            while (j <= row) {
                int vl2 = __riscv_vsetvl_e32m1(row - j + 1);
                // Load a[row*n + j] (row-major, lower triangle)
                vfloat32m1_t a_vec = __riscv_vle32_v_f32m1(a + row * n + j, vl2);
                // Load b[j..j+vl2-1]
                vfloat32m1_t b_part = __riscv_vle32_v_f32m1(b + j, vl2);
                // Multiply and reduce
                vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(a_vec, b_part, vl2);
                vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(prod, __riscv_vfmv_v_f_f32m1(0.0f, vl2), vl2);
                sum += __riscv_vfmv_f_s_f32m1_f32(red);
                j += vl2;
            }
            
            // Process columns j = row+1 .. n-1 (upper triangle, symmetric access)
            j = row + 1;
            while (j < n) {
                int vl2 = __riscv_vsetvl_e32m1(n - j);
                // Load a[j*n + row] (transposed access: column row of lower triangle)
                vfloat32m1_t a_vec = __riscv_vle32_v_f32m1(a + j * n + row, vl2);
                // Load b[j..j+vl2-1]
                vfloat32m1_t b_part = __riscv_vle32_v_f32m1(b + j, vl2);
                vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(a_vec, b_part, vl2);
                vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(prod, __riscv_vfmv_v_f_f32m1(0.0f, vl2), vl2);
                sum += __riscv_vfmv_f_s_f32m1_f32(red);
                j += vl2;
            }
            
            out[row] = sum;
        }
        
        i += vl;
    }
}
#include "harness.h"
