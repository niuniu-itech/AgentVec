#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 1
#define B_IS_MATRIX 0
#define C_IS_MATRIX 0
#define OUT_IS_MATRIX 0
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 1
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    size_t vl;
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        int j = 0;
        // Strip-mine the inner loop (j <= i)
        while (j <= i) {
            vl = vsetvl_e32m1(i - j + 1);
            vfloat32m1_t vec_a = vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t vec_b = vle32_v_f32m1(&b[j], vl);
            acc = vfmv_f_s_f32m1_f32(vfredusum_vs_f32m1_f32m1(vfmul_vv_f32m1(vec_a, vec_b, vl), vfmv_s_f_f32m1_f32(0.0f, vl), vl));
            j += vl;
        }
        out[i] = acc;
    }
}
#include "harness.h"
