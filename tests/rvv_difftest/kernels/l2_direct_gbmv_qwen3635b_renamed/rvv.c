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
    (void)c;
    
    // For each row i, compute out[i] = sum_{j=max(0,i-2)}^{min(n-1,i+2)} a[i*n + j] * b[j]
    // We process in strip-mined fashion with vsetvl
    
    int vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    
    for (int i = 0; i < n; i++) {
        int j_start = i - 2;
        if (j_start < 0) j_start = 0;
        int j_end = i + 2;
        if (j_end >= n) j_end = n - 1;
        
        int count = j_end - j_start + 1;
        
        // Load the band of b values
        // We need to load b[j_start..j_end] which is a contiguous segment
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + j_start, count);
        
        // Load the corresponding row segment from a[i*n + j_start .. i*n + j_end]
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i * n + j_start, count);
        
        // Multiply and reduce
        vfloat32m1_t vprod = __riscv_vmul_vv_f32m1(va, vb, count);
        float result = __riscv_vfredosumsu_f_f32m1_f32(vprod, vzero, count);
        
        out[i] = result;
    }
}
#include "harness.h"
