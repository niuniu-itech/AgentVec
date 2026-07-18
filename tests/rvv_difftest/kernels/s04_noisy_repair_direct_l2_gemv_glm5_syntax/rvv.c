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
#define BANDED_A_RADIUS -1
#include <riscv_vector.h>

static inline float av_hsum_f32m1(vfloat32m1_t v, size_t vl) {
    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t r = __riscv_vfredusum_vs_f32m1_f32m1(v, z, vl);
    return __riscv_vfmv_f_s_f32m1_f32(r);
}

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        const float *a_row = a + (size_t)i * n;
        float acc = 0.0f;
        int j = 0;
        for (; j < n;) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(n - j));
            vfloat32m1_t va = __riscv_vle32_v_f32m1(a_row + j, vl);
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j, vl);
            vfloat32m1_t vp = __riscv_vfmul_vv_f32m1(va, vb, vl);
            acc += av_hsum_f32m1(vp, vl);
            j += (int)vl;
        }
        out[i] = acc;
    }
}
#include "harness.h"
