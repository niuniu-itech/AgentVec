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
    (void)c;
    
    // Strip-mining loop for vector-length agnostic implementation
    for (int i = 0; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        
        // Process each row of the banded matrix multiplication
        // For row i, we compute sum_{j=max(0,i-2)}^{min(n-1,i+2)} a[i*n + j] * b[j]
        
        // We need to handle the banded structure carefully.
        // The band width is 5 (indices i-2 to i+2), but we must clamp to [0, n-1].
        
        // Since the band is small (at most 5 elements), we can process multiple rows
        // in a vectorized manner, but the dependency on b[j] with varying offsets makes
        // this tricky. Let's think about this differently.
        
        // Actually, for each output element out[i], we need a dot product of a small
        // window of row i of A with the entire vector b. The window has at most 5 elements.
        
        // A simpler approach: since the band width is small and fixed, we can unroll
        // or use scalar operations for the small dot products, but vectorize across
        // multiple rows.
        
        // Let's process vl rows at a time. For each row i in [start, start+vl-1],
        // we compute the dot product with b using the band constraints.
        
        // Since the band is only 5 wide, we can load the relevant b elements and
        // multiply. But the b elements needed vary by row.
        
        // Alternative: Since n might be large but the band is small, let's just
        // do the computation row by row but vectorize the inner product if possible.
        // However, with only 5 elements, vectorization of the inner product isn't
        // beneficial. Instead, let's vectorize across rows.
        
        // For each row i, out[i] = sum_{j=max(0,i-2)}^{min(n-1,i+2)} a[i*n+j] * b[j]
        
        // Let's handle this by processing each row individually but using vector
        // operations where beneficial. Given the small band width, we'll compute
        // each row's result using a small loop, but we can vectorize the outer loop
        // if we structure it properly.
        
        // Actually, the most straightforward vector-length-agnostic approach for
        // this banded matrix-vector multiply with a small band is to process each
        // row separately since the band width is constant and small.
        
        for (int row = 0; row < vl; row++) {
            int i = i_start + row;
            if (i >= n) break;
            
            int j_start = i - 2;
            if (j_start < 0) j_start = 0;
            int j_end = i + 2;
            if (j_end >= n) j_end = n - 1;
            
            float sum = 0.0f;
            for (int j = j_start; j <= j_end; j++) {
                sum += a[i * n + j] * b[j];
            }
            out[i] = sum;
        }
    }
}
#include "harness.h"
