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
        float acc = 0.0f;
        int j = 0;
        int vl;
        // Process the strictly lower triangle (j < i)
        // A[i, j] is stored in a[i*n + j]
        while (j < i) {
            vl = vsetvl_e32m1(i - j);
            vfloat32m1_t vec_a = vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vec_b = vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_prod = vfmul_vv_f32m1(vec_a, vec_b, vl);
            vfloat32m1_t vec_sum = vfredusum_vs_f32m1_f32m1(vec_prod, vfmv_s_f_f32m1(0.0f, vl), vl);
            acc += vfmv_f_s_f32m1_f32(vec_sum);
            j += vl;
        }
        // Process the diagonal and upper triangle (j >= i)
        // A[j, i] is stored in a[j*n + i]
        while (j < n) {
            vl = vsetvl_e32m1(n - j);
            vfloat32m1_t vec_a = vle32_v_f32m1(&a[j * n + i], vl);
            vfloat32m1_t vec_b = vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t vec_prod = vfmul_vv_f32m1(vec_a, vec_b, vl);
            vfloat32m1_t vec_sum = vfredusum_vs_f32m1_f32m1(vec_prod, vfmv_s_f_f32m1(0.0f, vl), vl);
            acc += vfmv_f_s_f32m1_f32(vec_sum);
            j += vl;
        }
        out[i] = acc;
    }
}
#include "harness.h"
