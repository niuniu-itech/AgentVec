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
        float sum = 0.0f;
        int j = 0;
        vfloat32m1_t vsum = vfmv_v_f32_m_v32m1(vfloat32m1_t){0}, vb, va;
        int vl;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            va = vle32_v_f32m1(a + i * n + j, vl);
            vb = vle32_v_f32m1(b + j, vl);
            vsum = vfmacc_vv_f32m1_m(vl, vsum, va, vb);
            j += vl;
        }
        // Extract the accumulated sum from vsum
        // Since we used m1, we need to reduce the vector to a scalar
        // vsum currently holds the partial sums in its lanes
        // We need to sum all lanes of vsum
        float result = 0.0f;
        for (int k = 0; k < 1; k++) { // m1 has 1 lane per element, but vsum is m1 type
            // Actually, vfmacc with m1 accumulates into the single lane of m1
            // So vsum[0] should contain the final result
            // Let's use a reduction approach that works for any vl
        }
        // Better approach: use vfredosum to reduce the vector
        // But we need to handle the case where vl might be 0 or small
        // Let's redo this more carefully
        
        // Reset and use proper reduction
        vfloat32m1_t vpartial = vfmv_v_f32_m_v32m1(vfloat32m1_t){0};
        j = 0;
        while (j < n) {
            vl = __riscv_vsetvl_e32m1(n - j);
            if (vl > 0) {
                va = vle32_v_f32m1(a + i * n + j, vl);
                vb = vle32_v_f32m1(b + j, vl);
                vpartial = vfmacc_vv_f32m1_m(vl, vpartial, va, vb);
            }
            j += vl;
        }
        
        // Reduce vpartial to a scalar using vfredosum
        // vfredosum_vf_f32m1_f32 reduces a vfloat32m1 to a float
        result = vfredosum_vf_f32m1_f32(vpartial, 0.0f);
        out[i] = result;
    }
}
#include "harness.h"
