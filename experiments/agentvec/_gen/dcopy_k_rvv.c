#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include "common.h"

int dcopy_k_rvv(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y) {
    if (n <= 0) return 0;
    vfloat64m8_t vx;
    if (inc_x == 1 && inc_y == 1) {
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
            vl = __riscv_vsetvl_e64m8(n); vx = __riscv_vle64_v_f64m8(x, vl); __riscv_vse64_v_f64m8(y, vx, vl);
        }
    } else {
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m8(n); vx = __riscv_vlse64_v_f64m8(x, sx, vl); __riscv_vsse64_v_f64m8(y, sy, vx, vl);
        }
    }
    return 0;
}
