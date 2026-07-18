#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float32_t acc = 0.0f;
    int i = 0;
    while (vl > 0) {
        vfloat32m1_t va = vld_v_f32m1(a + i, vl);
        vfloat32m1_t vb = vld_v_f32m1(b + i, vl);
        vfloat32m1_t vc = vld_v_f32m1(c + i, vl);
        
        // Perform element-wise operations and accumulate max
        // Note: The evidence shows a reduction (red.max.f32) over the input 'a'
        // However, the function signature takes a, b, c and outputs to 'out'.
        // The evidence is extremely sparse and contradictory to a standard vector kernel.
        // It suggests loading 'a', reducing max, and storing to out[0].
        // But it also loads b and c in the signature.
        // Given the instruction "Migrate this source-free GPU evidence", and the evidence
        // only uses 'a' for the reduction, we must interpret the intent.
        // The evidence: ld a[i]; red.max.f32 acc,a[i]; st out[0],acc
        // This implies a global reduction of max(a) stored to out[0].
        // However, the function signature has b and c.
        // Let's assume the evidence is incomplete or misleading regarding b and c,
        // or that b and c are unused in this specific "agentvec" logic which seems to be
        // a max-reduction of 'a'.
        
        // Vectorized max reduction of 'a'
        vfloat32m1_t vmax = va;
        for (int j = 1; j < vl; j++) {
             // Manual reduction loop for simplicity in vlen-agnostic code without intrinsics for full reduction
             // Or use vredmax_vf if available. RVV 1.0 has vredmax_vf.
        }
        
        // Using RVV 1.0 reduction intrinsic
        float max_val = __riscv_vredmax_vf_f32m1_f32(vmax, acc);
        acc = max_val;
        
        i += vl;
        vl = __riscv_vsetvl_e32m1(n - i);
    }
    
    // Store the result to out[0]
    *out = acc;
}
#include "harness.h"
