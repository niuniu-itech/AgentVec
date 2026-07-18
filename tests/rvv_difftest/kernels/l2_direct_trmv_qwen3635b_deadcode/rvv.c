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
    
    // We need to compute y[i] = sum_{j=0}^{i} a[i*n + j] * b[j] for i in [0, n)
    // This is a lower-triangular matrix-vector multiply.
    // Since each output element depends on a different row of the lower-triangular matrix,
    // and the number of non-zero elements per row varies (1 to n), we need to handle this carefully.
    // 
    // A straightforward vectorization approach:
    // For each row i, we compute the dot product of row i of A (elements a[i*n] to a[i*n+i]) with b[0..i].
    // The challenge is that the length of the dot product varies with i.
    //
    // We can use strip-mining with vsetvl to process chunks. However, since the dependency structure
    // is inherently sequential across rows (each row's computation is independent), we can vectorize
    // within each row's dot product.
    //
    // But note: the problem asks for vector-length agnostic code using __riscv_vsetvl_e32m1.
    // The simplest correct approach that preserves semantics is to process element by element or
    // use vector operations where possible.
    //
    // Let's think about this differently. We can vectorize the computation across multiple output elements
    // if we structure it properly. However, since each y[i] depends on a different number of terms,
    // a pure vector approach per-row is tricky.
    //
    // Actually, let's reconsider: we can process multiple rows at once if we pad or handle the varying lengths.
    // But the cleanest vector-length agnostic approach for this specific pattern:
    //
    // For each i, compute the dot product. We can vectorize the inner loop for each i.
    // Since n can be large, we want to vectorize the inner dot product computation.
    //
    // Let's use a strip-mining approach on the outer loop, processing multiple i values at once.
    // For a given chunk of rows, we can compute each row's dot product using vector operations.
    //
    // Simpler approach: process one row at a time, but vectorize the inner dot product.
    // For row i, we need to compute sum_{j=0}^{i} a[i*n+j] * b[j].
    // We can load up to vl floats from a[i*n + j] and b[j], multiply, and accumulate.
    
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        int j = 0;
        
        // Vectorize the inner loop for row i
        // The number of elements to process is i+1 (j from 0 to i)
        int remaining = i + 1;
        
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m1(remaining);
            
            // Load vector from a[i*n + j]
            vfloat32m1_t va = vle32_v_f32m1(a + i * n + j, vl);
            // Load vector from b[j]
            vfloat32m1_t vb = vle32_v_f32m1(b + j, vl);
            
            // Multiply and accumulate
            vfloat32m1_t vm = vmul_vv_f32m1(va, vb, vl);
            
            // We need to reduce this vector to a scalar. 
            // Since we're using m1, we can use vreduce_add or manually reduce.
            // Let's accumulate into s by reducing the vector.
            for (size_t k = 0; k < vl; k++) {
                s += vm[k];
            }
            
            j += (int)vl;
            remaining -= (int)vl;
        }
        
        out[i] = s;
    }
}
#include "harness.h"
