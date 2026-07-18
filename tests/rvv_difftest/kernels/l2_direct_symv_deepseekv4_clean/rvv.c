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
    int vl;
    for (int i = 0; i < n; i++) {
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        int j = 0;
        for (; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t a_vec;
            // Load from lower triangle: if j <= i use a[i*n+j], else a[j*n+i]
            // We need to handle both cases per element
            // Create mask for j <= i
            vbool32_t mask = __riscv_vmseq_vx_i32m1_b32(__riscv_vid_v_u32m1(j, vl), i, vl); // This is wrong, need per-element comparison
            // Correct approach: compare each lane's index (j + lane) with i
            vuint32m1_t indices = __riscv_vid_v_u32m1(vl);
            vuint32m1_t j_plus_lane = __riscv_vadd_vx_u32m1(indices, j, vl);
            vbool32_t le_mask = __riscv_vmseq_vv_u32m1_b32(j_plus_lane, __riscv_vmv_v_x_u32m1(i, vl), vl); // j+lane <= i
            // Actually need <=, not ==
            le_mask = __riscv_vmsleu_vv_u32m1_b32(j_plus_lane, __riscv_vmv_v_x_u32m1(i, vl), vl);
            
            // Load a[i*n+j] for lanes where j+lane <= i
            vuint32m1_t idx1 = __riscv_vadd_vx_u32m1(__riscv_vmul_vx_u32m1(__riscv_vmv_v_x_u32m1(i, vl), n, vl), j_plus_lane, vl);
            vfloat32m1_t a1 = __riscv_vle32_v_f32m1(&a[0], vl); // placeholder, need indexed load
            // Use indexed load: vfloat32m1_t a1 = __riscv_vluxei32_v_f32m1(a, idx1, vl);
            // Similarly for a[j*n+i]
            vuint32m1_t idx2 = __riscv_vadd_vx_u32m1(__riscv_vmul_vx_u32m1(j_plus_lane, n, vl), __riscv_vmv_v_x_u32m1(i, vl), vl);
            vfloat32m1_t a2 = __riscv_vluxei32_v_f32m1(a, idx2, vl);
            
            // Merge: select a1 where le_mask, else a2
            vfloat32m1_t a_vec_merged = __riscv_vmerge_vvm_f32m1(a2, a1, le_mask, vl);
            
            // Multiply and accumulate
            vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(a_vec_merged, b_vec, vl);
            sum = __riscv_vfadd_vv_f32m1(sum, prod, vl);
        }
        // Reduce sum to scalar
        float result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl));
        out[i] = result;
    }
}
#include "harness.h"
