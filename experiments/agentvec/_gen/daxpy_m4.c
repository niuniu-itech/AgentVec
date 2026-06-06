#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include "common.h"

int daxpy_k_rvv(BLASLONG n, BLASLONG d0, BLASLONG d1, double da,
           double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *d, BLASLONG d2) {
    if (n <= 0 || da == 0.0) return 0;
    vfloat64m4_t vx, vy;
    if (inc_x == 1 && inc_y == 1) {
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
            vl = __riscv_vsetvl_e64m4(n);
            vx = __riscv_vle64_v_f64m4(x, vl); vy = __riscv_vle64_v_f64m4(y, vl);
            vy = __riscv_vfmacc_vf_f64m4(vy, da, vx, vl); __riscv_vse64_v_f64m4(y, vy, vl);
        }
    } else {
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m4(n);
            vx = __riscv_vlse64_v_f64m4(x, sx, vl); vy = __riscv_vlse64_v_f64m4(y, sy, vl);
            vy = __riscv_vfmacc_vf_f64m4(vy, da, vx, vl); __riscv_vsse64_v_f64m4(y, sy, vy, vl);
        }
    }
    return 0;
}
