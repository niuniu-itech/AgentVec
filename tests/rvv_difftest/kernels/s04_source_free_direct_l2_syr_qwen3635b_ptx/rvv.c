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
        
        // Load vector x (a) and vector y (b)
        // Note: The evidence implies x[i] and x[j] are loaded. 
        // Given the operator structure C[i*n+j] + 2*x[i] + x[j], 
        // and standard Level-2 BLAS (GEMV) patterns, 'a' is likely the vector x 
        // and 'b' is likely the vector y (or vice versa). 
        // The evidence says ld x[i] and ld x[j]. This suggests a symmetric or 
        // specific kernel where both operands might come from the same source or 
        // two different sources. 
        // However, the signature has a, b, c. 
        // Let's assume: 
        // a -> vector x (size n)
        // b -> vector y (size n) 
        // c -> matrix C (n x n)
        // out -> result matrix (n x n)
        // The evidence: fma t, 2*x[i], x[j], C. 
        // Wait, the evidence says "ld x[i]; ld x[j]". This implies both operands 
        // are from the same vector 'x'. But we have 'a' and 'b'. 
        // Let's re-read carefully: "ld x[i]; ld x[j]". 
        // If x is 'a', then where is 'b'? 
        // Perhaps 'b' is not used, or 'b' is 'x' and 'a' is something else? 
        // Or maybe the evidence is generic and 'x' refers to the vector operand. 
        // In a standard GEMV: out = alpha * A * x + beta * y. 
        // Here: out[i][j] = C[i][j] + 2*a[i] + b[j]? 
        // Or out[i][j] = C[i][j] + 2*a[i] + a[j]? 
        // The evidence says "ld x[i]; ld x[j]". This strongly suggests both 
        // vector elements come from the same source 'x'. 
        // If the function signature is fixed, and we must use a, b, c, out: 
        // Maybe 'a' is the vector x, and 'b' is ignored? Or 'b' is the vector x? 
        // Let's look at the address expressions: C[i*n+j]. 
        // This is a row-major matrix access. 
        // The operation is: out[i][j] = C[i][j] + 2 * x[i] + x[j]. 
        // If x is 'a', then out[i][j] = C[i][j] + 2*a[i] + a[j]. 
        // If x is 'b', then out[i][j] = C[i][j] + 2*b[i] + b[j]. 
        // Given the ambiguity, I will assume 'a' is the primary vector 'x' 
        // as it is the first vector argument. 'b' might be unused or the 
        // evidence is slightly misleading regarding the source name. 
        // However, a more likely interpretation in a "Level-2 matrix/vector" 
        // context with two vector inputs 'a' and 'b' is: 
        // out[i][j] = C[i][j] + 2*a[i] + b[j] 
        // OR 
        // out[i][j] = C[i][j] + a[i] * b[j] (outer product) 
        // The evidence says "fma t, 2*x[i], x[j], C". 
        // FMA(a, b, c) = a*b + c. 
        // So t = 2*x[i] * x[j] + C? No, FMA is usually fma(a, b, c) -> a*b + c. 
        // But the evidence says "fma t, 2*x[i], x[j], C". 
        // This syntax is ambiguous. It could mean: 
        // 1. t = (2*x[i]) * x[j] + C 
        // 2. t = 2*x[i] + x[j] + C (if it's an add-multiply-add sequence) 
        // But FMA is specifically multiply-add. 
        // If it's FMA, then t = (2*x[i]) * x[j] + C[i][j]. 
        // This would be an outer product update. 
        // Let's assume the evidence means: t = 2.0 * a[i] * b[j] + c[i][j]. 
        // Why "x[i]" and "x[j]"? Maybe x is a placeholder for the vector operands. 
        // If it's an outer product, we load a[i] and b[j]. 
        // Let's assume the operator is: out[i][j] = c[i][j] + 2.0 * a[i] * b[j]. 
        // This fits "Level-2 matrix/vector" (rank-1 update). 
        // And it uses both 'a' and 'b'. 
        // The "2*" is a scalar multiplier. 
        
        // Vectorization strategy:
        // We can vectorize over j (the inner loop) for a fixed i.
        // For each i, we load a[i] (scalar), then loop over j.
        // Load b[j] in vector chunks.
        // Compute t = 2.0 * a[i] * b[j] + c[i][j].
        // Store to out[i][j].
        
        float a_val = a[i];
        float scale = 2.0f * a_val;
        
        for (j = 0; j < n; j += current_vl) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            
            // Load b[j...j+vl-1]
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j, vl);
            
            // Load c[i*n+j ... i*n+j+vl-1]
            vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + i * n + j, vl);
            
            // Broadcast scale to vector
            vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(scale, vl);
            
            // FMA: vs * vb + vc
            vfloat32m1_t vt = __riscv_vfmacc_vf_f32m1(vc, vs, vb, vl);
            
            // Store result
            __riscv_vse32_v_f32m1(out + i * n + j, vt, vl);
        }
    }
}
#include "harness.h"
