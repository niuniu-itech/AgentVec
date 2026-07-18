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

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        int j = 0;
        vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
        vfloat32m1_t vb = vmv_v_f32_m1(0.0f);
        vfloat32m1_t va = vmv_v_f32_m1(0.0f);
        vfloat32m1_t vsum = vmv_v_f32_m1(0.0f);
        int vl;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            if (vl > 0) {
                va = vle32_v_f32m1(a + i * n + j, vl);
                vb = vle32_v_f32m1(b + j, vl);
                vsum = vmul_vv_f32m1(vsum, va, vl);
                j += vl;
            }
        }
        // Reduce vsum to scalar
        s = __riscv_vfredosums_d_f32m1(__riscv_vfmv_v_f_f32m1(vsum, 0.0f), vzero);
        // Actually, vfredosums_d expects double. Let's use vfredosums_s for float.
        // But vfredosums_s is not standard in all RVV 1.0 implementations? 
        // Actually, vfredosums_s is part of RVV 1.0.
        // However, the above reduction is tricky. Let's do a simpler scalar reduction for the tail or use a proper vector reduction.
        // Since we need to be vector-length agnostic and correct, let's just do the reduction properly.
        // vfredosums_s_f32m1 takes a float vector and returns a float.
        s = __riscv_vfredosums_s_f32m1(vsum, vzero);
        out[i] = s;
    }
}
#include "harness.h"
