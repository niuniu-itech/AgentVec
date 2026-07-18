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
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        int j = 0;
        while (j < n) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b + j, vl);
            vfloat32m1_t vec_a;
            // Load the appropriate row/column elements from the symmetric matrix
            if (j + vl <= i) {
                // All elements in this chunk are from row i (lower triangle)
                vec_a = __riscv_vle32_v_f32m1(a + i * n + j, vl);
            } else if (j >= i) {
                // All elements in this chunk are from column i (upper triangle, transposed)
                vec_a = __riscv_vle32_v_f32m1(a + j * n + i, vl);
            } else {
                // Mixed: part below diagonal (row i), part above (column i)
                int split = i - j;
                vfloat32m1_t vec_low = __riscv_vle32_v_f32m1(a + i * n + j, split);
                vfloat32m1_t vec_high = __riscv_vle32_v_f32m1(a + (j + split) * n + i, vl - split);
                // Combine using a mask
                vbool32_t mask = __riscv_vmseq_vx_i32m1_b32(__riscv_vid_v_u32m1(vl), split, vl);
                vec_a = __riscv_vmerge_vvm_f32m1(vec_low, vec_high, mask, vl);
            }
            vfloat32m1_t vec_prod = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vec_prod, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl));
            j += vl;
        }
        out[i] = sum;
    }
}
#include "harness.h"
