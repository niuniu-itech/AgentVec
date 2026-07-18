#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    float min_val = a[0];
    
    if (n > 1) {
        int vl = __riscv_vsetvl_e32m1(n - 1);
        int i = 1;
        
        while (vl > 0) {
            vfloat32m1_t va = vle32_v_f32m1(a + i, vl);
            vfloat32m1_t vmin = vminv_f32m1_m(vl, va, min_val);
            min_val = vmin;
            
            i += vl;
            vl = __riscv_vsetvl_e32m1(n - i);
        }
    }
    
    out[0] = min_val;
}
#include "harness.h"
