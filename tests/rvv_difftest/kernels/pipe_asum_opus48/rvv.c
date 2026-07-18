/* saxpy RVV 1.0 (VLA, strip-mined): out[i] = 2*a[i] + b[i] */
#include <riscv_vector.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)c;
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
#ifdef INJECT_BUG
        vfloat32m1_t vr = __riscv_vfmacc_vf_f32m1(vb, 2.5f, va, vl); /* deliberate error */
#else
        vfloat32m1_t vr = __riscv_vfmacc_vf_f32m1(vb, 2.0f, va, vl); /* vb + 2*va */
#endif
        __riscv_vse32_v_f32m1(out + i, vr, vl);
    }
}
#include "harness.h"
