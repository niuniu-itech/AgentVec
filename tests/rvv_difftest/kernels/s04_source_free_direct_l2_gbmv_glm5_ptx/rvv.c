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
    int i;
    for (i = 0; i < n; i++) {
        int j_start = (i - 2 > 0) ? i - 2 : 0;
        int j_end   = (i + 2 < n - 1) ? i + 2 : n - 1;
        int len = j_end - j_start + 1;

        if (len <= 0) {
            out[i] = 0.0f;
            continue;
        }

        float acc = 0.0f;
        int j = j_start;

        // Vector strip-mining for the inner dot product
        while (j <= j_end) {
            int vl = vsetvl_e32m1(j_end - j + 1);

            vfloat32m1_t v_a = vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t v_b = vle32_v_f32m1(&b[j], vl);

            vfloat32m1_t v_fma = vfmacc_vf_f32m1(v_a, 1.0f, v_b, vl);

            // Horizontal sum of the FMA results
            vfloat32m1_t v_sum = vfredusum_vs_f32m1_f32m1(v_fma, vfmv_s_f_f32m1(0.0f, 1), vl);
            acc += vfmv_f_s_f32m1_f32(v_sum);

            j += vl;
        }

        out[i] = acc;
    }
}
#include "harness.h"
