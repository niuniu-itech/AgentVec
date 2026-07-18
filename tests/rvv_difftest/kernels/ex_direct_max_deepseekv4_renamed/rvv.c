#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    // Initialize reduction vector with first element
    float max_val = a[0];
    
    // Process remaining elements in vector-length agnostic way
    int i = 1;
    while (i < n) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec = __riscv_vle32_v_f32m1(&a[i], vl);
        
        // Broadcast current max to vector for comparison
        vfloat32m1_t max_vec = __riscv_vfmv_v_f_f32m1(max_val, vl);
        
        // Find new maximums in this chunk
        vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(max_vec, vec, vl);
        vfloat32m1_t new_max = __riscv_vfmv_v_f_f32m1(max_val, vl);
        new_max = __riscv_vmerge_vvm_f32m1(new_max, vec, mask, vl);
        
        // Reduce vector to scalar
        vfloat32m1_t red = __riscv_vfredmax_vs_f32m1_f32m1(new_max, new_max, vl);
        max_val = __riscv_vfmv_f_s_f32m1_f32(red);
        
        i += vl;
    }
    
    out[0] = max_val;
}
#include "harness.h"
