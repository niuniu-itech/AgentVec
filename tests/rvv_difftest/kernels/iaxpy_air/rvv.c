#include <riscv_vector.h>
#include <stdint.h>
#define DT int32_t
#define DT_IS_FLOAT 0
#define REDUCE 0

void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)a; (void)b; (void)c;
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vint32m1_t v0 = __riscv_vle32_v_i32m1(a + i, vl);
        vint32m1_t v1 = __riscv_vmul_vx_i32m1(v0, 3, vl);
        vint32m1_t v2 = __riscv_vle32_v_i32m1(b + i, vl);
        vint32m1_t v3 = __riscv_vsub_vv_i32m1(v1, v2, vl);
        __riscv_vse32_v_i32m1(out + i, v3, vl);
    }
}
#include "harness.h"
