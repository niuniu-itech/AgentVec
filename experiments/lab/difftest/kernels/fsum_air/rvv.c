#include <riscv_vector.h>
#include <stdint.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1

void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)a; (void)b; (void)c;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl);
    }
    __riscv_vse32_v_f32m1(out, acc, 1);
}
#include "harness.h"
