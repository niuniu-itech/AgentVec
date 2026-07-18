#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t va, vb, vc, vout;
    vfloat32m1_t vtwo = vfmv_v_f_f32m1(2.0f);
    
    while (vl > 0) {
        va = vle32_v_f32m1(a, vl);
        vb = vle32_v_f32m1(b, vl);
        vc = vle32_v_f32m1(c, vl);
        
        // The evidence shows: out[i] = a[i] * 2.0
        // Note: b and c are loaded but not used in the evidence's arithmetic.
        // We follow the evidence strictly: mul.f32 t, a[i], 2.0; st out[i], t
        vout = vmul_vf_f32m1(va, vtwo, vl);
        
        vse32_v_f32m1(out, vout, vl);
        
        a += vl;
        b += vl;
        c += vl;
        out += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
}
#include "harness.h"
