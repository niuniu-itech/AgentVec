#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include "common.h"

int dscal_k_rvv(BLASLONG n, BLASLONG d0, BLASLONG d1, double da,
           double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *d, BLASLONG d2) {
    if (n <= 0) return 0;
    vfloat64m8_t vx;
    if (inc_x == 1) {
        for (size_t vl; n > 0; n -= vl, x += vl) {
            vl = __riscv_vsetvl_e64m8(n);
            vx = __riscv_vle64_v_f64m8(x, vl); vx = __riscv_vfmul_vf_f64m8(vx, da, vl); __riscv_vse64_v_f64m8(x, vx, vl);
        }
    } else {
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {
            vl = __riscv_vsetvl_e64m8(n);
            vx = __riscv_vlse64_v_f64m8(x, sx, vl); vx = __riscv_vfmul_vf_f64m8(vx, da, vl); __riscv_vsse64_v_f64m8(x, sx, vx, vl);
        }
    }
    return 0;
}
