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
    for (int i = 0; i < n; i++) {
        float xi = a[i];
        int row_off = i * n;
        int j = 0;
        int vl;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t v_c = __riscv_vle32_v_f32m1(&c[row_off + j], vl);
            vfloat32m1_t v_xj = __riscv_vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t v_t = __riscv_vfmacc_vf_f32m1(v_c, 2.0f * xi, v_xj, vl);
            __riscv_vse32_v_f32m1(&out[row_off + j], v_t, vl);
            j += vl;
        }
    }
}
#include "harness.h"
