#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    if (n <= 0) {
        out[0] = 0.0f;
        return;
    }
    
    float min_val = a[0];
    int remaining = n - 1;
    const float *ptr = a + 1;
    
    while (remaining > 0) {
        vfloat32m1_t vl = __riscv_vsetvl_e32m1(remaining);
        if (vl <= 0) break;
        
        vfloat32m1_t va = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t vmin = __riscv_vmin_vf_f32m1(va, min_val, vl);
        
        // Reduce the vector to find the minimum in this chunk
        // We need to reduce vmin to a single scalar
        // Use a reduction approach: pairwise min then scalar
        vfloat32m1_t reduced;
        if (vl == 1) {
            reduced = va;
        } else {
            // For v1, v2, v4, v8, v16, v32, v64 with m1
            // We can use __riscv_vmin_vv to reduce pairwise
            // But we need to reduce to a single element.
            // Let's use a loop-based reduction within the vector or use min across lanes.
            // Actually, rvv has no direct "reduce min" intrinsic for m1 in all cases easily.
            // Let's use a different approach: accumulate min using scalar comparison after loading.
            
            // Alternative: load and compare element by element if vl is small, but that defeats vectorization.
            // Better: use __riscv_vmin_vv_f32m1 with itself to reduce, but that requires knowing the reduction tree.
            // Simplest correct approach for arbitrary vl with m1:
            // Since m1 has 1 LMUL, vl elements. We can reduce by repeatedly applying vmin_vv.
            
            // Let's just do a scalar loop for the reduction of the vector chunk, which is fine since it's small.
            // Or better: use the fact that we can extract and compare.
            
            // Actually, let's use a simpler strategy: process in chunks and maintain a running min.
            // For each chunk, we can find the min of the chunk by reducing the vector.
            
            // Let's use a straightforward reduction:
            vfloat32m1_t temp = va;
            // Reduce temp to a single element by pairwise min
            // This is tricky without knowing vl exactly at compile time.
            // Let's use a different tactic: compare each element of va with min_val using a mask and select.
            
            // Use vmin_vf to compare each element with min_val, then pick the smallest result.
            // vmin_vf returns the element-wise min. The smallest of those is the min of the chunk vs min_val.
            // But we still need to reduce vmin_vf result.
            
            // Let's just extract the vector and do a scalar reduction for the chunk.
            // This is acceptable for correctness and simplicity.
            float chunk_min = min_val;
            for (int i = 0; i < vl; i++) {
                float val = __riscv_vmv_x_f32m1_f32(temp, i);
                if (val < chunk_min) chunk_min = val;
            }
            min_val = chunk_min;
        }
        
        ptr += vl;
        remaining -= vl;
    }
    
    out[0] = min_val;
}
#include "harness.h"
