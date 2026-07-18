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
    (void)b;
    const float *A = c;
    float *O = out;
    const float *x = a;

    // Precompute 2.0f
    const float two = 2.0f;

    // We need to compute out[i*n + j] = A[i*n + j] + 2.0f * x[i] * x[j]
    // This is a rank-1 update: A + 2*x*x^T

    // Strip-mining loop
    for (int i = 0; i < n; i += 1) {
        // For each row i, we compute the outer product contribution
        // out[i*n + j] = A[i*n + j] + 2.0f * x[i] * x[j] for j = 0..n-1
        
        // Load x[i] as a scalar
        float xi = x[i];
        float xi_scaled = two * xi;

        // Process the row i with vectorization over j
        int j = 0;
        for (; j < n; j += 1) {
            // We can vectorize over j, but since we're doing a rank-1 update,
            // we need to load x[j] for each j. Let's use strip-mining over j.
        }
    }

    // Better approach: strip-mine over the entire n*n matrix in chunks
    // But the structure is row-major, so let's do strip-mining over columns for each row,
    // or better yet, strip-mine the entire output in VL chunks.

    // Let's use a single strip-mining loop over the entire n*n elements.
    // out[k] = A[k] + 2.0f * x[k/n] * x[k%n]
    
    // This is tricky because x[k/n] and x[k%n] change with k.
    // Let's do it row by row with vectorization over columns.

    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float xi_scaled = two * xi;
        
        int j = 0;
        while (j < n) {
            // Determine the number of elements to process in this chunk
            int vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load x[j:j+vl]
            vfloat32m1_t vx = vlm_v_f32m1(x + j, vl);
            
            // Load A[i*n + j:j+vl]
            vfloat32m1_t vA = vlm_v_f32m1(A + i * n + j, vl);
            
            // Compute xi_scaled * vx for each element
            // We need to broadcast xi_scaled to a vector
            vfloat32m1_t vxi = vmv_v_f32m1(xi_scaled, vl);
            
            // Multiply: vxi * vx
            vfloat32m1_t vprod = vmul_vv_f32m1(vxi, vx, vl);
            
            // Add to A
            vfloat32m1_t vout = vadd_vv_f32m1(vA, vprod, vl);
            
            // Store result
            vse1_v_f32m1(O + i * n + j, vout, vl);
            
            j += vl;
        }
    }
}
#include "harness.h"
