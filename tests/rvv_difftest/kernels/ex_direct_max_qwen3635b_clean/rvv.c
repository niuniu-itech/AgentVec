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
        vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
        
        // Find max in this chunk
        vfloat32m1_t vmax = va;
        for (int j = 1; j < vl; j++) {
            vfloat32m1_t vj = vle32_v_f32m1(a + i + j, 1);
            vmax = vmax > vj ? vmax : vj;
        }
        
        // Reduce to scalar max
        float chunk_max = vmax[0];
        for (int j = 1; j < vl; j++) {
            if (vmax[j] > chunk_max) {
                chunk_max = vmax[j];
            }
        }
        
        if (chunk_max > max_val) {
            max_val = chunk_max;
        }
    }
    
    out[0] = max_val;
}
#include "harness.h"
