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

static inline float av_hsum_f32m1(vfloat32m1_t v, size_t vl) {
    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t r = __riscv_vfredusum_vs_f32m1_f32m1(v, z, vl);
    return __riscv_vfmv_f_s_f32m1_f32(r);
}

static inline float av_dot_f32(const float *x, const float *y, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y + i, vl);
        vfloat32m1_t vp = __riscv_vfmul_vv_f32m1(vx, vy, vl);
        acc += av_hsum_f32m1(vp, vl);
        i += (int)vl;
    }
    return acc;
}

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        int lo = i - 2; if (lo < 0) lo = 0;
        int hi = i + 2; if (hi >= n) hi = n - 1;
        out[i] = av_dot_f32(a + (size_t)i * n + lo, b + lo, hi - lo + 1);
    }
}
#include "harness.h"
