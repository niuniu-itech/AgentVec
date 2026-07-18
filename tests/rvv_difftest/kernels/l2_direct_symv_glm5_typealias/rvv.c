#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 1
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; ) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            vbool32_t mask = __riscv_vmsle_vx_i32m1_b32(__riscv_vid_v_i32m1(vl), i - j, vl);
            vfloat32m1_t va = __riscv_vle32_v_f32m1_mask(mask, &a[i * n + j], __riscv_vle32_v_f32m1(&a[j * n + i], vl), vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            s = __riscv_vfmv_f_s_f32m1(__riscv_vfredusum_vs_f32m1_f32m1(__riscv_vfmul_vv_f32m1(va, vb, vl), __riscv_vfmv_s_f_f32m1(s, vl), vl));
            j += vl;
        }
        out[i] = s;
    }
}
#include "harness.h"
