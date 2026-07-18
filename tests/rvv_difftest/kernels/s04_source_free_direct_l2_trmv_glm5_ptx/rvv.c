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
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        int j = 0;
        int vl;
        // Strip-mine the inner loop (j <= i) using RVV vector-length-agnostic pattern
        for (; j <= i - (i % 4); j += vl) {
            vl = vsetvl_e32m1(i - j + 1);
            vfloat32m1_t v_L = vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t v_x = vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t v_acc = vle32_v_f32m1(&c[j], vl);
            vfloat32m1_t v_res = vfmacc_vf_f32m1(v_acc, acc, v_L, vl);
            vse32_v_f32m1(&out[j], v_res, vl);
        }
        // Handle the remaining tail elements
        if (j <= i) {
            vl = vsetvl_e32m1(i - j + 1);
            vfloat32m1_t v_L = vle32_v_f32m1(&a[i * n + j], vl);
            vfloat32m1_t v_x = vle32_v_f32m1(&b[j], vl);
            vfloat32m1_t v_acc = vle32_v_f32m1(&c[j], vl);
            vfloat32m1_t v_res = vfmacc_vf_f32m1(v_acc, acc, v_L, vl);
            vse32_v_f32m1(&out[j], v_res, vl);
        }
    }
}
#include "harness.h"
