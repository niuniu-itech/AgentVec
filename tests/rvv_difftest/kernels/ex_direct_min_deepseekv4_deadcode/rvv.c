#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>
#include <math.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    // Initialize the minimum with the first element
    float s = a[0];
    
    // Vectorized reduction to find minimum
    int i = 1;
    for (; i + __riscv_vsetvl_e32m1(n - i) <= n; ) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        // Compute vector minimum with broadcast of current scalar min
        vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(s, vl);
        vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(vec_a, vec_s, vl);
        // Update scalar min using masked reduction
        vfloat32m1_t vec_min = __riscv_vfmin_vv_f32m1(vec_a, vec_s, vl);
        // Extract the minimum from the vector (reduction)
        float min_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vec_min, __riscv_vfmv_v_f_f32m1(s, vl), vl));
        s = min_val;
        i += vl;
    }
    // Handle remaining elements (if any)
    for (; i < n; i++) {
        if (a[i] < s) s = a[i];
    }
    out[0] = s;
}
#include "harness.h"
