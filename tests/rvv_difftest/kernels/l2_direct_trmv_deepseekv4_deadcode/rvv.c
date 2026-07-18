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
    // Strip-mine over rows i = 0..n-1
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        // Process columns j = 0..i in vector chunks
        int j = 0;
        while (j <= i) {
            int vl = __riscv_vsetvl_e32m1(i - j + 1);
            // Load a[i*n + j .. i*n + j+vl-1] (row i, columns j..j+vl-1)
            vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i * n + j], vl);
            // Load b[j .. j+vl-1]
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[j], vl);
            // Multiply and reduce into scalar accumulator
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(va, vb, vl);
            acc = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(acc, vl), vl));
            j += vl;
        }
        out[i] = acc;
    }
}
#include "harness.h"
