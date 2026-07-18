#include <riscv_vector.h>
#include <math.h>
#include <stdint.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b; (void)c;
    size_t vl;
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + i, vl);
        vfloat32m1_t vout = __riscv_vfmul_vv_f32m1(va, vb, vl);
        vout = __riscv_vfadd_vv_f32m1(vout, vc, vl);
        __riscv_vse32_v_f32m1(out + i, vout, vl);
    }
}