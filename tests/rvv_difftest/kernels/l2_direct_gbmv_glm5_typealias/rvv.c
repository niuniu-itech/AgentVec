#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    int i = 0;
    while (i < n) {
        int32_t avl = n - i;
        int vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        int lo = i - 2;
        if (lo < 0) lo = 0;
        int hi = i + 2;
        if (hi >= n) hi = n - 1;

        for (int j = lo; j <= hi; j++) {
            vuint32m1_t vidx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(vl), i, vl);
            vbool32_t vmask = __riscv_vmsltu_vx_u32m1_b32(vidx, (uint32_t)n, vl);

            const float *a_row = a + j * n + i;
            vfloat32m1_t va = __riscv_vle32_v_f32m1_m(vmask, a_row, vl);

            float bj = b[j];
            vfloat32m1_t vb = __riscv_vfmv_v_f_f32m1(bj, vl);

            vsum = __riscv_vfmacc_vf_f32m1_m(vmask, vsum, bj, va, vl);
        }

        __riscv_vse32_v_f32m1(out + i, vsum, vl);
        i += vl;
    }
}
#include "harness.h"
