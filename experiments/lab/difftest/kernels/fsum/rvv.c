/* fsum RVV: unordered vector reduction => legal FP reassociation.
 * Differs from the sequential scalar sum in low bits, which is exactly why
 * SPR uses a float ULP/rtol tier rather than bit-exact equality. */
#include <riscv_vector.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl); /* unordered reduce */
    }
    __riscv_vse32_v_f32m1(out, acc, 1); /* store reduced lane 0 */
}
#include "harness.h"
