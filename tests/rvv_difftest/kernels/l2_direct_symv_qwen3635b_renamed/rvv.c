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
    int vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    
    for (int i = 0; i < n; i += vl) {
        int current_vl = __riscv_vsetvl_e32m1(n - i);
        
        // Load b elements for the current chunk
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, current_vl);
        
        // Initialize output vector to zero
        vfloat32m1_t vout = vzero;
        
        // For each row in the current chunk, compute dot product with lower triangle
        for (int row = 0; row < current_vl; row++) {
            int idx = i + row;
            // For row idx, we need sum_{j=0}^{idx} a[idx*n + j] * b[j]
            // But b[j] for j < i is not in the current vl window of vb
            // We need to handle the full dot product
            
            // Actually, let's think about this differently.
            // y[i] = sum_{j=0}^{n-1} A[i][j] * x[j]
            // Since A is symmetric and stored as lower triangle in row-major:
            // A[i][j] = a[i*n + j] if j <= i, else a[j*n + i]
            
            // For vectorization, we can process one row at a time or use a different approach.
            // Let's do one row per iteration of the outer loop for simplicity with strip mining.
            
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                float val;
                if (j <= idx) {
                    val = a[idx * n + j];
                } else {
                    val = a[j * n + idx];
                }
                sum += val * b[j];
            }
            out[idx] = sum;
        }
    }
}
#include "harness.h"
