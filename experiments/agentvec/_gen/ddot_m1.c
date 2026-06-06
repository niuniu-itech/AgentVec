#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include "common.h"

double ddot_k_rvv(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y) {
    if (n <= 0) return 0.0;
    size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t acc = __riscv_vfmv_v_f_f64m1(0.0, vlmax), vx, vy;
    if (inc_x == 1 && inc_y == 1) {
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
            vl = __riscv_vsetvl_e64m1(n);
            vx = __riscv_vle64_v_f64m1(x, vl); vy = __riscv_vle64_v_f64m1(y, vl);
            acc = __riscv_vfmacc_vv_f64m1(acc, vx, vy, vl);
        }
    } else {
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m1(n);
            vx = __riscv_vlse64_v_f64m1(x, sx, vl); vy = __riscv_vlse64_v_f64m1(y, sy, vl);
            acc = __riscv_vfmacc_vv_f64m1(acc, vx, vy, vl);
        }
    }
    double res;
    vfloat64m1_t r = __riscv_vfmv_v_f_f64m1(0.0, 1);
    r = __riscv_vfredusum_vs_f64m1_f64m1(acc, r, vlmax);
    __riscv_vse64_v_f64m1(&res, r, 1);
    return res;
}
