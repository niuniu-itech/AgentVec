#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl;
    vfloat32m1_t va, vb, vc, vt, vout;
    vfloat32m1_t vtwo = vfmv_v_f_f32m1(2.0f);

    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;

        va = vle32_v_f32m1(a + i, vl);
        vb = vle32_v_f32m1(b + i, vl);
        vc = vle32_v_f32m1(c + i, vl);

        // FFMA t, a[i], 2.0, b[i] -> t = a[i] * 2.0 + b[i]
        // Note: The evidence shows FFMA with a, 2.0, b. 
        // However, the function signature includes 'c' which is unused in the evidence.
        // We must follow the evidence logic: t = a * 2.0 + b
        vt = vfmadd_vv_f32m1(vfmul_vf_f32m1(va, vtwo, vl), vb, vl);

        vse32_v_f32m1(out + i, vt, vl);
    }
}
#include "harness.h"
