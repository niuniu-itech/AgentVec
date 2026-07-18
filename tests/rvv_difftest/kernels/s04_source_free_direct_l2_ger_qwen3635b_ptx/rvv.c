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
    int vn = __riscv_vsetvl_e32m1(n);
    
    for (i = 0; i < n; i += vn) {
        int vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;

        // Load vector x (a) and y (b)
        // Note: The evidence implies x is indexed by i (row) and y by j (col).
        // However, standard Level-2 GEMV/GER operations usually involve a vector and a matrix.
        // The address expression C[i*n+j] suggests a 2D layout.
        // The operation fma t, 2*x[i], y[j], C suggests:
        // out[i][j] = 2 * a[i] * b[j] + c[i][j]
        // This is a rank-1 update scaled by 2, or a scaled outer product addition.
        
        // Since RVV is vector-length agnostic and we are doing strip mining,
        // we need to handle the 2D nature. The inner loop over j is the vectorizable dimension.
        // The outer loop over i is the strip-mining dimension.
        
        // Let's re-interpret: The evidence says "for row i, col j".
        // If we vectorize over j (columns), then for a fixed i:
        // x_val = a[i]
        // We load a vector of b[j] for j in [0, vn)
        // We load a vector of c[i*n+j] for j in [0, vn)
        // Compute t = 2 * x_val * b[j] + c[i*n+j]
        // Store to out[i*n+j]
        
        float x_val = a[i];
        float two_x = 2.0f * x_val;
        
        // Vector load b
        vfloat32m1_t vb = vle32_v_f32m1(b + i * 0, vl); // Wait, b is indexed by j, not i.
        // The evidence says ld y[j]. So b is the vector y.
        // But b is passed as a pointer. Is b a vector of length n?
        // "ld x[i]" -> a is indexed by i.
        // "ld y[j]" -> b is indexed by j.
        // "ld C[i*n+j]" -> c is the matrix.
        
        // So for a fixed i, we iterate j.
        // x_val = a[i] is a scalar.
        // y_vec = b[0...vl-1] is a vector.
        // c_vec = c[i*n ... i*n+vl-1] is a vector.
        
        vfloat32m1_t vy = vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = vle32_v_f32m1(c + i * n, vl);
        
        // fma t, 2*x[i], y[j], C
        // t = 2 * x_val * vy + vc
        vfloat32m1_t vt = vfmacc_vf_f32m1(vc, two_x, vy);
        
        // st out[i*n+j], t
        vse32_v_f32m1(out + i * n, vt, vl);
    }
}
#include "harness.h"
