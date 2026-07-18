#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float acc = 0.0f;
    
    while (vl > 0) {
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = vle32_v_f32m1(c, vl);
        
        // Compute absolute value of a[i]
        vfloat32m1_t abs_a = vfabs_v_f32m1(va, vl);
        
        // Accumulate sum of absolute values
        acc += vfredosum_vs_f32m1_f32_m1(abs_a, acc, vl);
        
        // Store result to out[0] (scalar output from reduction)
        // Note: The evidence suggests a single scalar output 'out[0]'
        // based on STG.E out[0],acc. We write the accumulated sum.
        if (vl == n || vl == __riscv_vsetvl_e32m1(n - (n - vl))) {
            // Final iteration or single chunk
            // Since acc is scalar, we just write it once at the end
            // But to be safe with strip mining, we write only in the last iteration
            // However, vfredosum accumulates into acc. We need to ensure we write
            // the final accumulated value.
        }
        
        a += vl;
        b += vl;
        c += vl;
        out += 0; // out is scalar output
        
        vl = __riscv_vsetvl_e32m1(n - (n - vl)); // Update vl for next chunk
        // Actually, standard strip mining pattern:
        // vl = __riscv_vsetvl_e32m1(remaining)
        // But here we need to track remaining.
    }
    
    // The evidence shows a single accumulation and store.
    // Let's rewrite with proper strip mining loop structure.
}

// Corrected implementation with proper strip mining
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int remaining = n;
    float acc = 0.0f;
    
    while (remaining > 0) {
        int vl = __riscv_vsetvl_e32m1(remaining);
        
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = vle32_v_f32m1(c, vl);
        
        // Compute absolute value of a[i]
        vfloat32m1_t abs_a = vfabs_v_f32m1(va, vl);
        
        // Accumulate sum of absolute values into acc
        acc = vfredosum_vs_f32m1_f32_m1(abs_a, acc, vl);
        
        a += vl;
        b += vl;
        c += vl;
        remaining -= vl;
    }
    
    *out = acc;
}
#include "harness.h"
