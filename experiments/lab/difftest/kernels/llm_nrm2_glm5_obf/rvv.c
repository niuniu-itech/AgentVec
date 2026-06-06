#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#include <math.h>
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;

    // Obfuscated logic: if (n*n + 1) is even (always true), run a dummy loop.
    // This block has no side effects on the output.
    {
        long n_long = (long)n;
        if (((n_long * n_long + 1) & 1) == 0) {
            volatile double dummy = 0.0;
            for (long i = 0; i < n_long; ++i) {
                dummy += (double)i;
            }
            (void)dummy;
        }
    }

    // Vectorized reduction: sum += a[i] * a[i]
    float sum = 0.0f;
    int i = 0;
    size_t vl;

    // Use a large accumulator type to preserve precision similar to scalar float accumulation
    vfloat64m2_t vsum = __riscv_vle64_v_f64m2(&sum, 1);
    vsum = __riscv_vfmv_v_f_f64m2(0.0, 1); // Initialize with scalar 0

    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);

        // Load float vector
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);

        // Widening multiply: f32 * f32 -> f64
        vfloat64m2_t vprod = __riscv_vfwmul_vv_f64m2(va, va, vl);

        // Reduce the vector of products to a single scalar (f64)
        // We reduce into the existing vsum accumulator (which has vl=1)
        vfloat64m2_t vred = __riscv_vfredusum_vs_f64m2_f64m2(vprod, vsum, vl);

        // Update accumulator for next iteration
        vsum = vred;
        i += vl;
    }

    // Extract final sum
    sum = __riscv_vfmv_f_s_f64m2_f64(vsum);

    out[0] = sqrtf(sum);
}
#include "harness.h"
