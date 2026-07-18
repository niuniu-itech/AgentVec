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
    // Inferred Operator:
    // The assembly shows:
    // 1. Outer loop over 'row i'
    // 2. Inner loop over 'j' in range [i-2, i+2] (clipped to [0, n))
    // 3. Load A[i*n + j] (Row-major access to matrix A)
    // 4. Load x[j] (Vector B access)
    // 5. FFMA acc, A, x (Accumulate A[i*n+j] * B[j])
    // 6. Store acc to out[i]
    //
    // This is a sparse band-matrix-vector multiplication where the band width is 5
    // (indices i-2 to i+2). The matrix A is accessed row-wise, and vector B is accessed
    // sequentially. The result for row i is the dot product of the non-zero elements
    // in row i of A with the corresponding elements of B.

    const vfloat64m1_t vl = vsetvl_e32m1(n);
    const int64_t vl_int = (int64_t)vl;

    for (int64_t i = 0; i < n; i++) {
        float acc = 0.0f;
        
        // Determine the range [j_start, j_end) for the band [i-2, i+2]
        int64_t j_start = i - 2;
        if (j_start < 0) j_start = 0;
        
        int64_t j_end = i + 2;
        if (j_end >= n) j_end = n;

        // Process the band elements
        // Since the band width is small (max 5), we can use scalar loops or vectorize if needed.
        // However, RVV strip-mining is requested. Given the small, irregular access pattern 
        // relative to the whole vector, and the fact that j is sequential, we can vectorize 
        // the inner loop if the band is wide enough, but for a width of 5, scalar is likely 
        // more efficient. To strictly follow "vector-length-agnostic RVV strip-mining" 
        // and "source-free GPU evidence" implying a vectorized kernel structure, we will 
        // implement the inner accumulation using RVV if the segment is large, or scalar.
        // But wait, the prompt asks to migrate to RVV 1.0. The "strip-mining" usually applies 
        // to the outer loop or large inner loops. Here the inner loop is tiny.
        // Let's assume the "GPU evidence" implies a general pattern. If n is large, 
        // we might want to vectorize the outer loop if the operation was per-element.
        // But here, each output element depends on a small window of inputs.
        
        // Let's implement the inner loop with RVV for the sequential access of B and A.
        // We process the band [j_start, j_end) in chunks.
        
        int64_t j = j_start;
        while (j < j_end) {
            // Calculate remaining elements in this band
            int64_t rem = j_end - j;
            int64_t vl_inner = vsetvl_e32m1(rem);
            
            if (vl_inner > 0) {
                vfloat32m1_t va = vld_v_f32m1(&a[i * n + j], vl_inner);
                vfloat32m1_t vb = vld_v_f32m1(&b[j], vl_inner);
                vfloat32m1_t vprod = vmul_vv_f32m1(va, vb, vl_inner);
                
                // Reduce the product vector to a scalar sum
                // Since vl_inner is small (<=5), we can use a simple reduction or scalar loop.
                // For correctness and simplicity with small vectors:
                float sum = 0.0f;
                for (int64_t k = 0; k < vl_inner; k++) {
                    sum += vget_v_f32m1_x(vprod, k);
                }
                acc += sum;
            }
            j += vl_inner;
        }
        
        out[i] = acc;
    }
}
#include "harness.h"
