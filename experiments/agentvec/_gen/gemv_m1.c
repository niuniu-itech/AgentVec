#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include "common.h"

int dgemv_n_rvv(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha, double *a, BLASLONG lda, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *buffer) {
    if(n < 0) return(0);
    double *a_ptr, *x_ptr; BLASLONG i; vfloat64m1_t va, vy;
    if(inc_y == 1) {
        for (size_t vl; m > 0; m -= vl, y += vl, a += vl) {
            vl = __riscv_vsetvl_e64m1(m); a_ptr = a; x_ptr = x;
            vy = __riscv_vle64_v_f64m1(y, vl);
            for(i = 0; i < n; i++) { va = __riscv_vle64_v_f64m1(a_ptr, vl);
                vy = __riscv_vfmacc_vf_f64m1(vy, (alpha * (*x_ptr)), va, vl); a_ptr += lda; x_ptr += inc_x; }
            __riscv_vse64_v_f64m1(y, vy, vl);
        }
    } else {
        BLASLONG stride_y = inc_y * sizeof(double);
        for (size_t vl; m > 0; m -= vl, y += vl*inc_y, a += vl) {
            vl = __riscv_vsetvl_e64m1(m); a_ptr = a; x_ptr = x;
            vy = __riscv_vlse64_v_f64m1(y, stride_y, vl);
            for(i = 0; i < n; i++) { va = __riscv_vle64_v_f64m1(a_ptr, vl);
                vy = __riscv_vfmacc_vf_f64m1(vy, (alpha * (*x_ptr)), va, vl); a_ptr += lda; x_ptr += inc_x; }
            __riscv_vsse64_v_f64m1(y, stride_y, vy, vl);
        }
    }
    return(0);
}
