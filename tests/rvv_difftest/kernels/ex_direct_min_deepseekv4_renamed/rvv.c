#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <stdint.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;
    
    // Initialize with first element
    float min_val = a[0];
    
    // Vectorized reduction for elements 1..n-1
    int vl;
    int i = 1;
    for (; i + vl <= n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec = __riscv_vle32_v_f32m1(&a[i], vl);
        // Reduce within vector
        vfloat32m1_t min_vec = __riscv_vfmv_v_f_f32m1(min_val, vl);
        min_vec = __riscv_vfmin_vv_f32m1(min_vec, vec, vl);
        // Extract scalar minimum
        float vec_min = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vec, __riscv_vfmv_v_f_f32m1(__builtin_inff(), vl), vl));
        if (vec_min < min_val) min_val = vec_min;
    }
    // Handle remaining elements
    for (; i < n; i++) {
        if (a[i] < min_val) min_val = a[i];
    }
    
    out[0] = min_val;
}
#include "harness.h"
