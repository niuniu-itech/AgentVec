#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include "common.h"

double dasum_k_rvv(BLASLONG n, double *x, BLASLONG inc_x) {
    if (n <= 0 || inc_x <= 0) return 0.0;
    size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t acc = __riscv_vfmv_v_f_f64m4(0.0, vlmax), vx;
    if (inc_x == 1) {
        for (size_t vl; n > 0; n -= vl, x += vl) {
            vl = __riscv_vsetvl_e64m4(n); vx = __riscv_vle64_v_f64m4(x, vl);
            vx = __riscv_vfabs_v_f64m4(vx, vl); acc = __riscv_vfadd_vv_f64m4(acc, vx, vl);
        }
    } else {
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {
            vl = __riscv_vsetvl_e64m4(n); vx = __riscv_vlse64_v_f64m4(x, sx, vl);
            vx = __riscv_vfabs_v_f64m4(vx, vl); acc = __riscv_vfadd_vv_f64m4(acc, vx, vl);
        }
    }
    double res;
    vfloat64m1_t r = __riscv_vfmv_v_f_f64m1(0.0, 1);
    r = __riscv_vfredusum_vs_f64m4_f64m1(acc, r, vlmax);
    __riscv_vse64_v_f64m1(&res, r, 1);
    return res;
}
