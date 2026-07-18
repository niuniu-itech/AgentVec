#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 0
#define B_IS_MATRIX 0
#define C_IS_MATRIX 1
#define OUT_IS_MATRIX 1
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>
void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int i;
    for (i = 0; i < n; i++) {
        float xi = a[i];
        float two_xi = 2.0f * xi;
        int j = 0;
        size_t vl;
        while (j < n) {
            vl = vsetvl_e32m1(n - j);
            vfloat32m1_t v_y = vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t v_c = vle32_v_f32m1(&c[i * n + j], vl);
            vfloat32m1_t v_out = vfmacc_vf_f32m1(v_c, two_xi, v_y, vl);
            vse32_v_f32m1(&out[i * n + j], v_out, vl);
            j += vl;
        }
    }
}
#include "harness.h"
