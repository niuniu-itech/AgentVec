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
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        int j = 0;
        while (j < n) {
            int vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b + j, vl);
            
            // Load a[i*n+j] for j <= i, else a[j*n+i]
            // We need to compute the correct source address for each element
            // For j <= i: address = a + i*n + j
            // For j > i:  address = a + j*n + i
            // We'll load both possibilities and select using a mask
            vfloat32m1_t vec_a_lower = __riscv_vle32_v_f32m1(a + i * n + j, vl);
            vfloat32m1_t vec_a_upper = __riscv_vle32_v_f32m1(a + j * n + i, vl);
            
            // Create mask: j + vl_index <= i
            vbool32_t mask = __riscv_vmseq_vv_i32m1_b32(
                __riscv_vadd_vx_i32m1(__riscv_vid_v_i32m1(vl), j, vl),
                __riscv_vmv_v_x_i32m1(i, vl),
                vl
            );
            
            vfloat32m1_t vec_a = __riscv_vmerge_vvm_f32m1(vec_a_lower, vec_a_upper, mask, vl);
            
            vfloat32m1_t vec_prod = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vec_prod, 
                __riscv_vfmv_s_f_f32m1(__riscv_vundefined_f32m1(), 0.0f, vl), vl));
            
            j += vl;
        }
        out[i] = sum;
    }
}
#include "harness.h"
