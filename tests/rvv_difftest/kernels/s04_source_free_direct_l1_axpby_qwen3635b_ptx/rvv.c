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
    vfloat32m1_t vthree = vfmv_v_f_f32m1(3.0f);

    while (vl > 0) {
        va = vle32_v_f32m1(a, vl);
        vb = vle32_v_f32m1(b, vl);
        vc = vle32_v_f32m1(c, vl);
        
        // x = a[i] * 2.0
        vfloat32m1_t vx = vmul_vv_f32m1(va, vtwo, vl);
        
        // t = b[i] * 3.0 + x
        vfloat32m1_t vt = vfmacc_vv_f32m1(vx, vb, vthree, vl);
        
        // out[i] = t
        vse32_v_f32m1(out, vt, vl);
        
        a += vl;
        b += vl;
        c += vl;
        out += vl;
        n -= vl;
        vl = __riscv_vsetvl_e32m1(n);
    }
}
#include "harness.h"
