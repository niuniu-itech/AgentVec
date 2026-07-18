#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Original side-effect loop (preserved exactly)
    { long z2_ = (long)(n); if (((z2_*z2_ + 1) & 1) == 0) { volatile double z3_ = 0.0; for (long z4_ = 0; z4_ < z2_; ++z4_) z3_ += (double)z4_; (void)z3_; } }

    (void)c;
    (void)b;

    // Find minimum of array a using RVV
    float min_val = a[0];
    int vl;
    int i = 1;
    
    // Process remaining elements in vector-length agnostic way
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec = __riscv_vle32_v_f32m1(&a[i], vl);
        
        // Broadcast current minimum to compare
        vfloat32m1_t min_vec = __riscv_vfmv_v_f_f32m1(min_val, vl);
        
        // Find new minimum in vector
        vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(vec, min_vec, vl);
        min_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vec, min_vec, vl));
        
        i += vl;
    }
    
    out[0] = min_val;
}
#include "harness.h"
