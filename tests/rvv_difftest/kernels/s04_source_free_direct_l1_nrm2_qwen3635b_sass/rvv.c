#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    while (vl > 0) {
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = vle32_v_f32m1(c, vl);
        
        // The evidence sketch shows: FMUL t,a[i],a[i]; FADD.RED acc,t; sqrt.rn.f32 r,acc
        // This implies computing sqrt(a[i]^2) which is fabs(a[i]).
        // However, the function signature includes b and c which are unused in the sketch.
        // Given the prompt asks to migrate the *evidence* to RVV 1.0, and the evidence
        // only uses 'a' (LDG.E a[i]), we follow the evidence logic.
        // Note: The evidence sketch is extremely minimal and likely illustrative.
        // Standard interpretation of "agentvec" with a,b,c often implies a[i]*b[i] + c[i] or similar.
        // But the evidence explicitly says: FMUL t,a[i],a[i] -> t = a[i]*a[i]
        // FADD.RED acc,t -> acc = sum(t) (reduction)
        // sqrt.rn.f32 r,acc -> r = sqrt(acc)
        // STG.E out[0],r -> out[0] = r
        
        // This is a reduction operation producing a single scalar result stored at out[0].
        // It does not produce an output vector of length n.
        
        vfloat32m1_t va_sq = vmul_vv_f32m1(va, va, vl);
        vfloat32m1_t acc = va_sq;
        
        // Perform reduction sum
        float sum = __riscv_vfredosum_vs_f32m1_f32_m1(acc, 0.0f, vl);
        
        // Compute square root
        float result = sqrtf(sum);
        
        // Store result
        out[0] = result;
        
        // Since the output is a single scalar, we break after the first iteration
        // or handle the case where n > 0. The evidence stores to out[0].
        break;
    }
}
#include "harness.h"
