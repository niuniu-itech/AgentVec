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
    int vl = __riscv_vsetvl_e32m1(n);
    int i = 0;
    while (i < n) {
        if (vl == 0) break;
        // For each row k in the current vector chunk, we need to compute
        // out[k] = sum_{j=0}^{k} a[k*n + j] * b[j]
        // This is a triangular dot product per row.
        // Since each row has a different number of elements (k+1), we cannot
        // directly vectorize across rows with a single vdot or similar.
        // However, we can process each row individually using vector operations.
        // But the requirement is to use strip-mining with vsetvl.
        // Let's think about this differently.
        // The kernel computes y[i] = sum_{j=0}^{i} L[i][j] * x[j] for i=0..n-1
        // where L is lower triangular in row-major order.
        // L[i][j] is at a[i*n + j].
        
        // For a given row i, the number of non-zero elements is i+1.
        // We need to compute a dot product of length i+1.
        // Since i varies, we can't easily vectorize across different rows.
        
        // Alternative approach: Process element by element or in small groups.
        // But let's try to vectorize the inner loop for each row.
        
        // Actually, let's reconsider. The standard approach for triangular matvec
        // with RVV is to process each row. For row i, we have i+1 elements.
        // We can use vfmv.v.f and vmv.v.s to initialize accumulators, then
        // use vector load and multiply-add.
        
        // Let's process one row at a time within the strip-mined loop.
        // But that would mean vl=1 effectively for the outer loop.
        
        // Better approach: For each row i from 0 to n-1:
        //   Compute dot product of a[i*n : i*n+i+1] and b[0:i+1]
        //   Store result in out[i]
        
        // We can vectorize the dot product for each row if i+1 is large enough.
        // Let's process rows in groups where possible, but it's complex.
        
        // Simplest correct approach: Process each row individually.
        // For row i, load up to i+1 elements from a and b, multiply and accumulate.
        
        // Let me re-read the requirement. It says to use __riscv_vsetvl_e32m1 strip-mining.
        // This typically applies to the outer loop. Let's process multiple rows per VL.
        
        // For each row k in the current VL chunk:
        //   The number of elements to process is k+1.
        //   We need to compute sum_{j=0}^{k} a[k*n+j]*b[j]
        
        // Since the length varies per row, we handle each row separately.
        // Let's iterate through each row in the current VL chunk.
        
        for (int k = 0; k < vl; k++) {
            int row = i + k;
            if (row >= n) break;
            
            int len = row + 1; // Number of elements in this row's dot product
            float sum = 0.0f;
            
            // Process the dot product for this row
            // We can use vector operations for this
            int remaining = len;
            int offset_a = row * n;
            
            while (remaining > 0) {
                int current_vl = __riscv_vsetvl_e32m1(remaining);
                if (current_vl == 0) break;
                
                vfloat32m1_t va = vle32_v_f32m1(a + offset_a, current_vl);
                vfloat32m1_t vb = vle32_v_f32m1(b, current_vl);
                vfloat32m1_t vprod = vmul_vv_f32m1(va, vb, current_vl);
                
                // Reduce the product to a scalar sum
                // Use vfredusum to reduce
                float partial_sum = __riscv_vfredusum_vs_f32m1_f32_m1(vprod, __riscv_vfmv_v_f_f32m1(0.0f, current_vl), current_vl);
                sum += partial_sum;
                
                offset_a += current_vl;
                b += current_vl;
                remaining -= current_vl;
            }
            
            out[row] = sum;
        }
        
        i += vl;
        vl = __riscv_vsetvl_e32m1(n - i);
    }
}
#include "harness.h"
