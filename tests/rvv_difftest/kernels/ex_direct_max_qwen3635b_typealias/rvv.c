#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    float max_val = a[0];
    int i = 1;
    
    // Process in chunks using strip-mining
    for (; i < n; i += __riscv_vsetvl_e32m1(n - i)) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        if (vl <= 0) break;
        
        // Load vector
        vfloat32m1_t va = vle32_v_f32m1(&a[i], vl);
        
        // Find max in this chunk
        vfloat32m1_t vmax = va;
        for (int j = 1; j < vl; j++) {
            // We need to compare elements. Since we can't easily do element-wise max in a loop with intrinsics,
            // let's use a different approach: reduce the vector to find max.
        }
        
        // Actually, let's use vmax.vv or a reduction approach.
        // For vfloat32m1, we can use a sequential reduction within the vector or use vrgather/vmax.
        // A simpler approach: use a loop over the vector elements to find max.
        
        // Let's use a proper reduction:
        // First, get the max of the vector using a tree reduction or sequential.
        // Since vl can be up to 8 for m1, we can do a simple loop.
        
        float chunk_max = -1e30f;
        for (int j = 0; j < vl; j++) {
            float val = va[j];
            if (val > chunk_max) chunk_max = val;
        }
        
        if (chunk_max > max_val) {
            max_val = chunk_max;
        }
    }
    
    out[0] = max_val;
}
#include "harness.h"
