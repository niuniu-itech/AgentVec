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
    int i, j;
    int vl = __riscv_vsetvl_e32m1(n);
    
    for (i = 0; i < n; i += vl) {
        int current_vl = __riscv_vsetvl_e32m1(n - i);
        
        // Load vector elements from 'a' (x[i]) and 'b' (x[j] in evidence, mapped to b)
        // Evidence: LDG.E x[i], LDG.E x[j] -> a[i], b[j]
        // Operator: FFMA t, 2*x[i], x[j], C -> 2.0f * a[i] * b[j] + c[i*n+j]
        
        // We need to handle the 2D indexing. 
        // The evidence loops row i, col j. 
        // 'a' corresponds to x[i] (row vector), 'b' corresponds to x[j] (col vector).
        // 'c' is the matrix C.
        // 'out' is the result.
        
        // Since this is a Level-2 operator (GEMV-like or rank-1 update), 
        // and we are using strip mining, we can vectorize over the inner dimension (j) 
        // if we treat 'a' as a scalar broadcast or if 'a' is also a vector.
        // However, the evidence shows LDG.E x[i] inside the loop over i,j? 
        // No, "for row i,col j" implies nested loops.
        // LDG.E x[i] is likely outside the j loop or loaded once per i.
        // LDG.E x[j] is inside the j loop.
        // LDG.E C[i*n+j] is inside the j loop.
        
        // Let's restructure for RVV strip-mining over the inner dimension (j) for each row i.
        // But the function signature takes n, implying a square matrix or vector size.
        // Let's assume we vectorize over the inner loop j.
        
        for (i = 0; i < n; i++) {
            float a_val = a[i];
            float scale = 2.0f * a_val;
            
            int j = 0;
            int vl = __riscv_vsetvl_e32m1(n);
            
            while (j < n) {
                int current_vl = __riscv_vsetvl_e32m1(n - j);
                
                // Load b[j...j+vl-1]
                vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j, current_vl);
                
                // Load c[i*n + j ... i*n + j + vl - 1]
                vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + i * n + j, current_vl);
                
                // Broadcast a_val to vector
                vfloat32m1_t va = __riscv_vfmv_v_f_f32m1(a_val, current_vl);
                
                // FFMA: t = 2.0f * a[i] * b[j] + c[i*n+j]
                // __riscv_vfma_vv_f32m1(dest, src1, src2, vl)
                // We want: out = scale * b + c
                // scale is 2.0f * a[i]
                
                vfloat32m1_t t = __riscv_vfmul_vf_f32m1(vb, scale, current_vl);
                t = __riscv_vfma_vv_f32m1(t, vc, current_vl);
                
                // Store result
                __riscv_vse32_v_f32m1(out + i * n + j, t, current_vl);
                
                j += current_vl;
            }
        }
    }
}
#include "harness.h"
