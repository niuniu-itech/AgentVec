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
    (void)c;
    
    long vl;
    int i = 0;
    
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        // Process each row i to i+vl-1
        for (int row = i; row < i + (int)vl; row++) {
            float sum = 0.0f;
            
            // For row 'row', we compute dot product of row 'row' of A with b
            // A is stored such that a[row*n + col] for col <= row (lower triangle)
            // and a[col*n + row] for col > row (upper triangle, symmetric)
            
            // We need to handle the triangular access pattern.
            // For columns 0 to row: use a[row*n + col]
            // For columns row+1 to n-1: use a[col*n + row]
            
            // Let's split into two parts:
            // Part 1: col from 0 to min(row, i+vl-1) but we need to be careful with vectorization
            
            // Actually, let's think about this differently.
            // For a given row r, y[r] = sum_{c=0}^{n-1} A[r][c] * b[c]
            // where A[r][c] = a[r*n + c] if c <= r, else a[c*n + r]
            
            // To vectorize efficiently, we can process the lower part (c <= r) and upper part (c > r) separately.
            
            // Lower part: c from 0 to r
            int lower_end = row; // columns 0..row inclusive
            if (lower_end >= n) lower_end = n - 1;
            
            // Vectorize the lower part
            int col = 0;
            while (col <= lower_end) {
                long vl2 = __riscv_vsetvl_e32m1(lower_end - col + 1);
                if (vl2 > 0) {
                    vfloat32m1_t va = __riscv_vle32_v_f32m1(a + row * n + col, vl2);
                    vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + col, vl2);
                    vfloat32m1_t vprod = __riscv_vmul_vv_f32m1(va, vb, vl2);
                    sum += __riscv_vfredosum_vs_f32m1_f32_m1(vprod, __riscv_vfmv_v_f_f32m1(0.0f, vl2), vl2);
                }
                col += (int)vl2;
            }
            
            // Upper part: c from row+1 to n-1
            int upper_start = row + 1;
            if (upper_start < n) {
                int col2 = upper_start;
                while (col2 < n) {
                    long vl3 = __riscv_vsetvl_e32m1(n - col2);
                    if (vl3 > 0) {
                        // For columns col2..col2+vl3-1, A[row][c] = a[c*n + row]
                        // We need to gather these values
                        vfloat32m1_t va;
                        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + col2, vl3);
                        
                        // Gather a[c*n + row] for c in [col2, col2+vl3-1]
                        // This requires indexed/gather load which may not be directly available in basic RVV
                        // Let's use a different approach: precompute or use scalar for gather
                        
                        // Since gather is complex, let's handle upper part with a loop or use vle with offset calculation
                        // For simplicity and correctness, let's use scalar for the gather part if needed
                        // But let's try to vectorize by computing indices
                        
                        // Actually, let's just do scalar for the upper triangular part to ensure correctness
                        for (int c = col2; c < col2 + (int)vl3 && c < n; c++) {
                            sum += a[c * n + row] * b[c];
                        }
                    }
                    col2 += (int)vl3;
                }
            }
            
            out[row] = sum;
        }
        
        i += (int)vl;
    }
}
#include "harness.h"
