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
    (void)b;
    int i, j;
    vfloat32m1_t va;
    vfloat32m1_t vc;
    vfloat32m1_t vout;
    vfloat32m1_t vtwo = vfmv_v_f_f32m1(2.0f);
    int vl;
    int remaining;
    const float *a_ptr;
    const float *c_ptr;
    float *out_ptr;
    float a_i;
    float a_j;
    float c_val;
    float out_val;

    for (i = 0; i < n; i++) {
        a_i = a[i];
        for (j = 0; j < n; j++) {
            c_ptr = c + i * n + j;
            c_val = *c_ptr;
            a_j = a[j];
            out_val = c_val + 2.0f * a_i * a_j;
            out_ptr = out + i * n + j;
            *out_ptr = out_val;
        }
    }
}
#include "harness.h"
