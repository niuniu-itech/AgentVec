#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    float acc = 0.0f;
    vfloat32m1_t va, vb, vc;
    vfloat32m1_t vabs_a;
    vfloat32m1_t vzero = vmv_v_f32_m1(0.0f);
    
    while (vl > 0) {
        va = vle32_v_f32_m1(a, vl);
        vb = vle32_v_f32_m1(b, vl);
        vc = vle32_v_f32_m1(c, vl);
        
        vabs_a = vabs_v_f32_m1(va);
        
        // Accumulate absolute values of a
        // Since we need a scalar reduction, we can use a loop or intrinsic if available.
        // However, standard RVV 1.0 doesn't have a direct vredsum for m1 to scalar without mask handling.
        // We'll do a simple scalar reduction over the vector elements for correctness and portability.
        for (int i = 0; i < vl; i++) {
            acc += __riscv_vfmv_v_f_f32_m1(vabs_a, i);
        }
        
        a += vl;
        b += vl;
        c += vl;
        out += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
    
    // Store the accumulated result to out[0] as per the evidence
    out[0] = acc;
}
#include "harness.h"
