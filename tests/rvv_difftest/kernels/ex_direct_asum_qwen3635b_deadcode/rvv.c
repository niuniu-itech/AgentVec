#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float s = 0.0f;
    
    // Handle the scalar case or small n
    if (n <= 0) {
        out[0] = s;
        return;
    }
    
    // Use strip-mining with vsetvl for vector-length agnostic code
    int vl;
    vfloat32m1_t va;
    vfloat32m1_t vabs;
    vfloat32m1_t vsum;
    vfloat32m1_t vzero = vmv_v_f32m1(0.0f);
    
    // We need to accumulate absolute values. Since fabsf is not directly a vector intrinsic
    // for absolute value in a single step for all cases, we use the fact that for floats,
    // we can mask off the sign bit. However, rvv doesn't have a direct fabs intrinsic.
    // We can use vmsgtu_v_v or similar, but the standard way is to use the sign bit mask.
    // Actually, rvv has vfabs_v for floating point absolute value in some implementations,
    // but let's check: RVV 1.0 does have vfabs_v for floating point types.
    
    // Process in chunks
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;
        
        // Load vector
        va = vle32_v_f32m1(a + i, vl);
        
        // Compute absolute value
        vabs = vfabs_v_f32m1(va, vl);
        
        // Accumulate into a running sum vector
        // We need to accumulate across iterations. Let's use a scalar accumulator approach
        // with vector reduction at each step, or accumulate in a vector register.
        // Since we only have one output, let's accumulate in a vector and reduce at the end.
        // But we need to handle the accumulation across strip-mining iterations.
        
        // Option 1: Accumulate into a vector accumulator
        // We'll keep a vector accumulator and reduce at the end.
        // But we need to initialize it. Let's use a different approach:
        // accumulate into a scalar variable using vector reduction at each step.
        
        vfloat32m1_t vchunk_sum = vadd_vv_f32m1(vzero, vabs, vl);
        // Reduce the chunk to a scalar
        float chunk_sum = __riscv_vfredosum_vs_f32m1_f32_m1(vchunk_sum, vzero, vl);
        s += chunk_sum;
        
        i += vl;
    }
    
    out[0] = s;
}
#include "harness.h"
