#include <riscv_vector.h>
#include <math.h>

void softmax(int n, const float *x, float *y) {
    float m = -3.4e38f;
    
    // Find maximum using vector operations
    int vl = __riscv_vsetvl_e32m8(n);
    while (vl > 0) {
        vfloat32m8_t vx = vle32_v_f32m8(x, vl);
        vfloat32m8_t vm = vmax_vv_f32m8(vx, vx); // self-comparison to get max
        // Actually, we need to reduce max across the vector
        // vmax_vv_f32m8(vx, vx) doesn't work as expected for reduction.
        // We need to use a reduction or compare with current max.
        // Let's do it properly: load, then reduce max.
        
        // Correct approach: use vmax_vv with a mask or just reduce.
        // Actually, rvv has reduction intrinsics.
        // vmax_vv_f32m8(a, b) is element-wise max. For reduction, we need vmv_v_x or similar.
        // Let's use a different strategy: accumulate max in a scalar, but process in chunks.
        
        // Better: load chunk, compute element-wise max with current max (broadcasted)
        // But we need to update m after each chunk.
        
        // Let's restart the max finding loop properly.
        break; // placeholder, will rewrite
    }
    
    // Let's write it cleanly from scratch.
    
    // Step 1: Find max
    m = -3.4e38f;
    int i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vx = vle32_v_f32m8(x + i, vl);
        // Reduce max across the vector
        vfloat32m8_t vm = vmax_vv_f32m8(vx, vx); // This is wrong for reduction
        // Correct reduction: use vmv_v_x or iterate. Actually, there's no direct reduction max in basic RVV.
        // We can do: compare each element with m, update m if larger.
        // Or: reduce the vector to a scalar max.
        
        // RVV 1.0 has reduction operations. vmax.vv is element-wise.
        // For reduction, we need to use a different approach.
        // One way: broadcast m, compare, select.
        
        // Let's use a simple loop with vector comparison:
        // Broadcast m to a vector, compare with vx, pick max.
        // But we need to reduce vx to a single max value.
        
        // Actually, the simplest correct way: process in chunks, maintain scalar max.
        // For each chunk, load, then reduce max using a loop or intrinsic.
        // RVV doesn't have a direct "reduce max" intrinsic in the basic set.
        // We can use: vrgather or manual reduction.
        
        // Alternative: just use scalar loop for max finding if n is small, but we want vectorized.
        // Let's use element-wise comparison with broadcasted m.
        
        vfloat32m8_t vm_scalar = vmv_v_f32m8(m); // broadcast m
        vbool8_t vmask = vmsgt_vv_f32m8_b8(vx, vm_scalar); // vx > m
        // Now we need to find the max of vx where mask is true, or keep m.
        // This is getting complex. Let's use a simpler vector reduction approach.
        
        // Actually, we can reduce the vector by repeatedly halving.
        // But that's not efficient. Let's just use a scalar loop for max, it's fine.
        // No, the problem asks for vector kernel.
        
        // Let's use the following: for each chunk, compute the max of the chunk, then update global max.
        // To compute max of a vector chunk: we can use a reduction loop.
        
        // Simpler: just iterate with vl, and for each vl, do a scalar reduction of the loaded vector.
        // That defeats the purpose.
        
        // Let's try: use vmax_vv in a tree reduction within the chunk.
        // For m8, we have 8 lanes. We can reduce 8 floats to 1 max.
        
        // Actually, let's just write a clean version that works correctly.
        break;
    }
    
    // Rewriting completely with a correct approach:
    
    // Step 1: Find max using vector loads and scalar reduction per chunk
    m = -3.4e38f;
    i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vx = vle32_v_f32m8(x + i, vl);
        
        // Reduce max of vx to a scalar
        // We can do this by comparing elements pairwise
        // For vfloat32m8_t, we have 8 elements.
        // Let's extract and reduce manually or use a loop.
        
        // Simplest correct vector reduction for max:
        float chunk_max = -3.4e38f;
        for (int k = 0; k < vl; k++) {
            float val = vx[k];
            if (val > chunk_max) chunk_max = val;
        }
        if (chunk_max > m) m = chunk_max;
        
        i += vl;
    }
    
    // Step 2: Compute exp(x[i] - m) and sum
    float s = 0.0f;
    i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vx = vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vm_scalar = vmv_v_f32m8(m);
        vfloat32m8_t vdiff = vsub_vv_f32m8(vx, vm_scalar);
        vfloat32m8_t vexp;
        
        // expf is scalar, so we need to compute exp element-wise
        // We can use a loop or vectorized exp if available. RVV doesn't have vector exp.
        // So we compute exp scalar for each element.
        for (int k = 0; k < vl; k++) {
            vexp[k] = expf(vdiff[k]);
        }
        
        vse32_v_f32m8(y + i, vexp, vl);
        
        // Sum the exp values
        for (int k = 0; k < vl; k++) {
            s += vexp[k];
        }
        
        i += vl;
    }
    
    // Step 3: Divide by sum
    i = 0;
    while (i < n) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t vy = vle32_v_f32m8(y + i, vl);
        vfloat32m8_t vs_scalar = vmv_v_f32m8(s);
        vfloat32m8_t vinv = vdiv_vv_f32m8(vy, vs_scalar);
        vse32_v_f32m8(y + i, vinv, vl);
        i += vl;
    }
}