#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    (void)b;
    float s = 0.0f;
    int vl;
    vfloat32m1_t va, vb, vprod;
    vfloat32m1_t vsum = vmv_v_f_f32m1(0.0f, 1);

    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        va = vle32_v_f32m1(a + i, vl);
        vb = vle32_v_f32m1(b + i, vl);
        vprod = vmul_vv_f32m1(va, vb, vl);
        vsum = vfredosum_vs_f32m1_f32m1(vprod, vsum, vsum, vl);
    }

    // Extract the final sum from the vector register
    float result;
    result = vfredosum_vs_f32m1_f32m1(vsum, vsum, vsum, 1);
    // Actually, vfredosum reduces across lanes. After the loop, vsum holds the partial sum.
    // We need to reduce vsum to a scalar.
    // Let's use a different approach: accumulate into a scalar variable using vector reductions.
    
    // Reset and do it properly with scalar accumulation from vector reductions
    s = 0.0f;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        va = vle32_v_f32m1(a + i, vl);
        vb = vle32_v_f32m1(b + i, vl);
        vprod = vmul_vv_f32m1(va, vb, vl);
        // Reduce the vector product to a scalar and add to s
        float partial;
        partial = vfredosum_vs_f32m1_f32m1(vprod, vprod, vprod, vl);
        s += partial;
    }
    
    out[0] = s;
}
#include "harness.h"
