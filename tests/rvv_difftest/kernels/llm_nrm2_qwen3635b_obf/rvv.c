#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float sum = 0.0f;
    int i = 0;
    vfloat32m1_t vzero = vmv_v_f_f32m1(0.0f);
    vfloat32m1_t vsum = vmv_v_f_f32m1(0.0f);
    vfloat32m1_t va;
    vfloat32m1_t va_sq;
    float tmp[32]; // Max LMUL for m1 is 32 elements if ELEN=32, but we use strip mining so this buffer size is just for accumulation. 
                   // Actually, for m1, vlmax depends on VLEN. We accumulate into a scalar or a small vector.
                   // Since we need to sum all squares, let's use a scalar accumulator for simplicity and correctness across VLENS.
    
    // However, to be efficient and correct with RVV, we can accumulate in a vector register of type m1.
    // But since we don't know VLEN, we must use strip mining.
    // Let's accumulate into a scalar variable to avoid complex vector reduction logic that might be tricky with unknown VLEN.
    // Or, we can use vfredosum.
    
    // Let's use a scalar accumulator for the final sum to keep it simple and robust.
    // But wait, the prompt asks for RVV intrinsics. Using a scalar loop is not using RVV.
    // We should use RVV to compute the sum of squares.
    
    // Strategy:
    // 1. Load chunks of 'a' into vector registers.
    // 2. Square them.
    // 3. Reduce the vector to a scalar sum.
    // 4. Add to total sum.
    // 5. Repeat until done.
    
    // We need to handle the case where n is 0.
    if (n <= 0) {
        out[0] = 0.0f;
        return;
    }

    float total_sum = 0.0f;
    int vl;
    vfloat32m1_t v_a;
    vfloat32m1_t v_a_sq;
    vfloat32m1_t v_acc = vmv_v_f_f32m1(0.0f);
    
    // We will accumulate the sum of squares in a vector register of type m1.
    // At the end of each chunk, we reduce it to a scalar and add to total_sum.
    
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vector
        v_a = vle32_v_f32m1(a + i, vl);
        
        // Square the elements
        v_a_sq = vmul_vv_f32m1(v_a, v_a, vl);
        
        // Add to accumulator
        v_acc = vfredosum_vs_f32m1_f32m1(v_acc, v_a_sq, vl);
        
        // If this is the last chunk, we need to reduce the accumulator to a scalar
        if (i + vl >= n) {
            // Reduce the vector accumulator to a scalar
            // vfredosum reduces the vector to the first element of the accumulator register
            // We need to extract the first element.
            // Since we are using m1, the accumulator has vlmax elements.
            // We can use vse32 to store it to a temporary buffer and take the first element,
            // or use a reduction intrinsic if available.
            // vfredosum returns a vector. The first element is the sum.
            
            // To extract the first element from v_acc:
            // We can store it to a stack buffer and read the first element.
            float acc_buf[32]; // Max LMUL for m1 is 32
            __riscv_vse32_v_f32m1(acc_buf, v_acc, vl);
            total_sum += acc_buf[0];
        }
    }
    
    out[0] = sqrtf(total_sum);
}
#include "harness.h"
