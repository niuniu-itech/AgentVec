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
    int vl;
    vfloat32m1_t vmin;
    
    // Initialize vmin with the first element
    vmin = vfmv_v_f_f32m1(min_val, 1);
    
    // Process the rest of the array using strip-mining
    int i = 1;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl <= 0) break;
        
        vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
        vmin = vminv_vv_f32m1(vmin, va, vl);
        
        i += vl;
    }
    
    // Extract the minimum value from the vector register
    min_val = vmin[0];
    
    // Handle any remaining elements that might not have been covered
    // Actually, the strip-mining loop above should cover all elements from index 1 to n-1
    // But let's verify: if n=1, the loop doesn't execute, min_val stays a[0], which is correct.
    
    out[0] = min_val;
}
#include "harness.h"
