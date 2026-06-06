/* NAIVE FIXED-WIDTH translation of an AVX idiom (e.g. daxpy_microk_haswell:
 * %ymm = 256-bit = 8 floats, advance by a hardcoded lane count). This mimics what
 * a syntactic SIMD->RVV translation produces when it copies the source's fixed
 * width instead of recovering intent. It hardcodes 8 elements/iteration.
 * Correct only when the hardware VLEN matches the assumed 256-bit width. */
#include <riscv_vector.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define FIXED 8   /* assume 256-bit / 8 float lanes, like the AVX source */
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)c;
    int i = 0;
    for (; i + FIXED <= n; i += FIXED) {            /* advance by the ASSUMED width */
        size_t vl = __riscv_vsetvl_e32m1(FIXED);    /* clamps to vlmax if VLEN<256 */
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vr = __riscv_vfmacc_vf_f32m1(vb, 2.0f, va, vl);
        __riscv_vse32_v_f32m1(out + i, vr, vl);     /* writes only vl, but i += FIXED */
    }
    for (; i < n; i++) out[i] = 2.0f * a[i] + b[i]; /* scalar remainder */
}
#include "harness.h"
