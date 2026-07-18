#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    float min_val = __builtin_inff();
    int vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(&b[i], vl);
        vfloat32m1_t vec_c = __riscv_vle32_v_f32m1(&c[i], vl);
        
        // Compute a[i] + b[i] * c[i] (since no specific operation was given, 
        // we use a common pattern; adjust as needed)
        vfloat32m1_t result = __riscv_vfmacc_vv_f32m1(vec_a, vec_b, vec_c, vl);
        
        // Find minimum across vector
        vfloat32m1_t min_vec = __riscv_vfmv_v_f_f32m1(min_val, vl);
        min_vec = __riscv_vfmin_vv_f32m1(min_vec, result, vl);
        
        // Reduce to scalar
        float scalar_min = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(min_vec, min_vec, vl));
        if (scalar_min < min_val) min_val = scalar_min;
    }
    out[0] = min_val;
}
#include "harness.h"
