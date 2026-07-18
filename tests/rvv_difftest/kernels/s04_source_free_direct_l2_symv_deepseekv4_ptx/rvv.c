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
        float acc = 0.0f;
        int j = 0;
        for (; j < n; j += vl) {
            vl = __riscv_vsetvl_e32m8(n - j);
            vfloat32m8_t vec_x = __riscv_vle32_v_f32m8(&b[j], vl);
            vfloat32m8_t vec_a;
            // Select lower-triangle element A[max(i,j),min(i,j)]
            // For each j in the vector, compute the index into the packed lower-triangle
            // The lower triangle is stored row-major: A[row][col] where row >= col
            // Index = row*(row+1)/2 + col
            // Here row = max(i, j+offset), col = min(i, j+offset)
            // We need to compute this per element
            for (int k = 0; k < vl; k++) {
                int jk = j + k;
                int row = (i > jk) ? i : jk;
                int col = (i < jk) ? i : jk;
                int idx = row * (row + 1) / 2 + col;
                // Load single element and insert into vector
                vec_a = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                vec_a = __riscv_vfmv_s_f_f32m1_f32m8(vec_a, a[idx], k);
            }
            vfloat32m8_t vec_prod = __riscv_vfmul_vv_f32m8(vec_a, vec_x, vl);
            vfloat32m1_t vec_sum = __riscv_vfmv_s_f_f32m1_f32m8(__riscv_vfmv_v_f_f32m8(0.0f, vl), 0.0f, 0);
            vec_sum = __riscv_vfredusum_vs_f32m8_f32m1(vec_prod, vec_sum, vl);
            acc += __riscv_vfmv_f_s_f32m1_f32m1(vec_sum);
        }
        out[i] = acc;
    }
}
#include "harness.h"
