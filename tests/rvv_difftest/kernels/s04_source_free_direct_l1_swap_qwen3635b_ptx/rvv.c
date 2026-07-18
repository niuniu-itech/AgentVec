#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    int vl;
    vfloat32m1_t va, vb, vc, vout;
    vuint32m1_t vidx;
    int i = 0;

    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl == 0) break;

        va = vle32_v_f32m1(a + i, vl);
        vb = vle32_v_f32m1(b + i, vl);
        vc = vle32_v_f32m1(c + i, vl);

        // Based on evidence: st out[i], b[i]; st out[n+i], a[i]
        // Note: 'c' is loaded but not used in the evidence logic, 
        // but signature requires it. We ignore 'c' as per evidence.
        
        vse32_v_f32m1(out + i, vb, vl);
        vse32_v_f32m1(out + n + i, va, vl);

        i += vl;
    }
}
#include "harness.h"
