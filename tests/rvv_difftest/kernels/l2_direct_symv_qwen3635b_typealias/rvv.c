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
    int i = 0;
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
    
    for (i = 0; i < n; i++) {
        // Compute y[i] = sum_{j=0}^{n-1} A[i][j] * b[j]
        // A[i][j] = a[i*n + j] if j <= i, else a[j*n + i]
        
        // We need to compute the dot product of row i of A with vector b.
        // Row i of A has elements:
        //   j=0..i:   a[i*n + j]
        //   j=i+1..n-1: a[j*n + i]
        
        // Since we need to be vector-length agnostic and handle arbitrary n,
        // we'll process in chunks using strip mining.
        
        // For each element j, we need to fetch the appropriate element from a.
        // This is tricky because the access pattern for the "upper triangle" part
        // (j > i) accesses a[j*n + i], which is not contiguous in a simple way.
        
        // Let's think about this differently. The matrix A is symmetric.
        // A[i][j] = A[j][i].
        // For j <= i: A[i][j] = a[i*n + j] (stored in row-major lower triangle)
        // For j > i: A[i][j] = A[j][i] = a[j*n + i] (stored in row-major lower triangle as element at row j, col i)
        
        // So for a fixed i, as j varies:
        // - When j <= i: we access a[i*n + j], which is contiguous in memory (row i, cols 0..i)
        // - When j > i: we access a[j*n + i], which is strided by n (column i, rows i+1..n-1)
        
        // This mixed access pattern makes straightforward vectorization difficult.
        // However, we can still use vectorization for the contiguous part and handle
        // the strided part separately, or use a different approach.
        
        // Alternative: Process the entire computation using scalar operations for correctness,
        // but vectorize where possible. Given the complexity, let's use a simpler approach:
        // Vectorize the inner loop by handling the two cases.
        
        float sum = 0.0f;
        
        // Process j from 0 to n-1
        // We'll use strip mining for the entire inner loop
        
        int j = 0;
        int vl;
        
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            
            // For each element in this chunk, determine if j <= i or j > i
            // This depends on i, which is fixed for the outer loop.
            
            // Case 1: If the entire chunk has j <= i (i.e., j + vl - 1 <= i)
            // Case 2: If the entire chunk has j > i (i.e., j > i)
            // Case 3: Mixed case where some j <= i and some j > i
            
            if (j + vl - 1 <= i) {
                // All elements in this chunk have j <= i
                // Access a[i*n + j] which is contiguous
                vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
                vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
                float s = __riscv_vfredosumsf_vf_f32m1_f32(vprod, 0.0f, vl);
                sum += s;
            } else if (j > i) {
                // All elements in this chunk have j > i
                // Access a[j*n + i] which is strided by n
                // We need to load elements at offsets j*n + i for j in [j, j+vl-1]
                // These are at positions: j*n+i, (j+1)*n+i, ..., (j+vl-1)*n+i
                // Stride is n elements (n * sizeof(float) bytes)
                
                vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
                
                // Load strided elements from a
                // __riscv_vlse32_v_f32m1 loads with a stride
                vfloat32m1_t va = __riscv_vlse32_v_f32m1(&a[i], n, vl);
                
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
                float s = __riscv_vfredosumsf_vf_f32m1_f32(vprod, 0.0f, vl);
                sum += s;
            } else {
                // Mixed case: split the chunk
                int mid = i - j + 1; // number of elements where j <= i in this chunk
                if (mid > 0 && mid < vl) {
                    // Process first mid elements (j <= i)
                    vfloat32m1_t va1 = __riscv_vle32_v_f32m1(&a[i * n + j], mid);
                    vfloat32m1_t vb1 = __riscv_vle32_v_f32m1(&b[j], mid);
                    vfloat32m1_t vprod1 = __riscv_vfmul_vv_f32m1(va1, vb1, mid);
                    float s1 = __riscv_vfredosumsf_vf_f32m1_f32(vprod1, 0.0f, mid);
                    sum += s1;
                    
                    // Process remaining elements (j > i)
                    int rem = vl - mid;
                    vfloat32m1_t vb2 = __riscv_vle32_v_f32m1(&b[j + mid], rem);
                    vfloat32m1_t va2 = __riscv_vlse32_v_f32m1(&a[i], n, rem);
                    vfloat32m1_t vprod2 = __riscv_vfmul_vv_f32m1(va2, vb2, rem);
                    float s2 = __riscv_vfredosumsf_vf_f32m1_f32(vprod2, 0.0f, rem);
                    sum += s2;
                } else if (mid == 0) {
                    // Should not happen since j <= i and we're in the else branch
                    // Actually if mid == 0, it means j > i, which should be caught above
                    // If mid == vl, it means all j <= i, which should be caught above
                    // So this case shouldn't occur
                    int rem = vl;
                    vfloat32m1_t vb2 = __riscv_vle32_v_f32m1(&b[j], rem);
                    vfloat32m1_t va2 = __riscv_vlse32_v_f32m1(&a[i], n, rem);
                    vfloat32m1_t vprod2 = __riscv_vfmul_vv_f32m1(va2, vb2, rem);
                    float s2 = __riscv_vfredosumsf_vf_f32m1_f32(vprod2, 0.0f, rem);
                    sum += s2;
                } else if (mid == vl) {
                    // All j <= i, should be caught above
                    vfloat32m1_t va1 = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
                    vfloat32m1_t vb1 = __riscv_vle32_v_f32m1(&b[j], vl);
                    vfloat32m1_t vprod1 = __riscv_vfmul_vv_f32m1(va1, vb1, vl);
                    float s1 = __riscv_vfredosumsf_vf_f32m1_f32(vprod1, 0.0f, vl);
                    sum += s1;
                }
            }
            
            j += vl;
        }
        
        out[i] = sum;
    }
}
#include "harness.h"
