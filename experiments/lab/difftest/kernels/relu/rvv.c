/* AgentVec migration of the oneDNN CUDA leaky-ReLU kernel -> RVV VLA.
 * Recovered intent: leaky ReLU = max(x, slope*x) for slope in [0,1]. */
#include <riscv_vector.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c;
    const float slope = 0.01f;
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vs = __riscv_vfmul_vf_f32m1(vx, slope, vl);
        vfloat32m1_t vr = __riscv_vfmax_vv_f32m1(vx, vs, vl);
        __riscv_vse32_v_f32m1(out + i, vr, vl);
    }
}
#include "harness.h"
