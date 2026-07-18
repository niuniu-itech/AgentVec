#include <riscv_vector.h>
#include <stdint.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0

void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t v_b = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t v_c = __riscv_vle32_v_f32m1(c + i, vl);
        vfloat32m1_t v_tmp = __riscv_vfmul_vv_f32m1(v_a, v_b, vl);
        vfloat32m1_t v_result = __riscv_vfadd_vv_f32m1(v_tmp, v_c, vl);
        __riscv_vse32_v_f32m1(out + i, v_result, vl);
    }
}