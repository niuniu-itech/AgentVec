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
    int i, j;
    int vl = __riscv_vsetvl_e32m1(n);
    
    for (i = 0; i < n; i++) {
        float x_val = a[i];
        float scale = 2.0f * x_val;
        
        for (j = 0; j < n; j += vl) {
            int current_vl = __riscv_vsetvl_e32m1(n - j);
            
            vfloat32m1_t b_vec = __riscv_vle_v_f32m1(b + j, current_vl);
            vfloat32m1_t c_vec = __riscv_vle_v_f32m1(c + i * n + j, current_vl);
            
            // t = 2*x[i]*y[j] + C[i*n+j]
            // Since scale is scalar, we can use vfmadd.vf or vfmul.vf + vfadd.vf
            // Using vfmadd.vf: rd = rd + rs1 * rs2
            // We want: out = c_vec + scale * b_vec
            // vfmadd.vf(rd, rs1, rs2) -> rd = rd + rs1 * rs2
            // So: rd = c_vec, rs1 = scale, rs2 = b_vec
            
            vfloat32m1_t out_vec = __riscv_vfmadd_vf_f32m1(c_vec, scale, b_vec, current_vl);
            
            __riscv_vse_v_f32m1(out + i * n + j, out_vec, current_vl);
        }
    }
}
#include "harness.h"
