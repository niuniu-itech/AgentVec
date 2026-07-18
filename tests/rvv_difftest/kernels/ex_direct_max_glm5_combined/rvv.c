#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
#include <riscv_vector.h>

void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b;
    (void)c;

    float max_val = a[0];
    int i = 1;

    if (n > 1) {
        size_t vl = __riscv_vsetvl_e32m1(n - 1);
        vfloat32m1_t v_max = __riscv_vle32_v_f32m1(&a[1], vl);
        i += vl;

        while (i < n) {
            vl = __riscv_vsetvl_e32m1(n - i);
            vfloat32m1_t v_data = __riscv_vle32_v_f32m1(&a[i], vl);
            v_max = __riscv_vfmax_vv_f32m1(v_max, v_data, vl);
            i += vl;
        }

        vfloat32m1_t v_tmp = v_max;
        size_t curr_vl = __riscv_vsetvl_e32m1(vl);
        while (curr_vl > 1) {
            v_tmp = __riscv_vfmax_vv_f32m1(v_tmp, __riscv_vslidedown_vx_f32m1(v_tmp, 1, curr_vl), curr_vl);
            curr_vl = __riscv_vsetvl_e32m1(curr_vl >> 1);
        }
        max_val = __riscv_vfmv_f_s_f32m1(v_tmp);
    }

    out[0] = max_val;
}
#include "harness.h"
