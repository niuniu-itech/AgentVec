#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float acc = 0.0f;
    vfloat32m1_t va;
    vfloat32m1_t va2;
    vfloat32m1_t vsum;
    vfloat32m1_t vzero = vmv_v_f32m1(0.0f);
    
    while (vl > 0) {
        va = vle32_v_f32m1(a, vl);
        va2 = vmul_vv_f32m1(va, va, vl);
        vsum = vredsum_vs_f32m1_f32m1_e32m1(va2, vzero, vl);
        acc += vsum[0];
        a += vl;
        vl = __riscv_vsetvl_e32m1(n - (a - (const float *)0)); // Simplified logic for strip mining
        // Note: The evidence shows a single reduction over the whole vector, 
        // but strip mining requires accumulating partial results.
        // The evidence is ambiguous about whether 'a' is the only input or if b/c are used.
        // Given "ld a[i]; mul.f32 t,a[i],a[i]; red.add.f32 acc,t; sqrt.rn.f32 r,acc; st out[0],r"
        // It seems to compute sqrt(sum(a[i]^2)) for all i.
        // However, the function signature has a, b, c. The evidence ignores b and c.
        // We must follow the evidence's logic: sum of squares of 'a', then sqrt, store to out[0].
        // But wait, the evidence stores to out[0], implying a single output.
        // The function signature suggests element-wise or block-wise operation?
        // "agentvec" might imply a specific kernel.
        // Let's re-read carefully: "ld a[i]; mul.f32 t,a[i],a[i]; red.add.f32 acc,t; sqrt.rn.f32 r,acc; st out[0],r"
        // This computes one scalar result for the entire vector 'a'.
        // But the function takes 'n' elements.
        // If it's a single reduction, why strip mining?
        // Perhaps the evidence is simplified.
        // Let's assume the standard pattern for such a reduction with strip mining.
        
        // Actually, looking at the evidence again:
        // It loads a[i], squares it, reduces to acc, takes sqrt, stores to out[0].
        // This is a global reduction.
        // However, the prompt asks for "strip-mining" with "vsetvl".
        // This implies we should process chunks.
        // But the output is a single float at out[0].
        // So we accumulate the sum of squares across chunks.
        
        // Let's refine the loop.
    }
    
    // Reset for correct implementation
    float sum_sq = 0.0f;
    int remaining = n;
    int current_vl;
    
    while (remaining > 0) {
        current_vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t va = vle32_v_f32m1(a, current_vl);
        vfloat32m1_t va_sq = vmul_vv_f32m1(va, va, current_vl);
        
        // Reduce sum of squares
        // vredsum_vs_f32m1_f32m1_e32m1 returns a vector of length 1 (m1)
        vfloat32m1_t vred = vredsum_vs_f32m1_f32m1_e32m1(va_sq, vmv_v_f32m1(0.0f), current_vl);
        sum_sq += vred[0];
        
        a += current_vl;
        remaining -= current_vl;
    }
    
    float result = sqrtf(sum_sq);
    *out = result;
}
#include "harness.h"
