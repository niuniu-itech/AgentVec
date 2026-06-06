#include <riscv_vector.h>
void gemv(int N, const double *A, const double *x, double *y) {
    for (int i = 0; i < N; ++i) {
        const double *row = A + (size_t)i * (size_t)N;
        double sum = 0.0;
        for (int j = 0; j < N; ) {
            size_t vl = __riscv_vsetvl_e64m1((size_t)(N - j));
            vfloat64m1_t av = __riscv_vle64_v_f64m1(row + j, vl);
            vfloat64m1_t xv = __riscv_vle64_v_f64m1(x + j, vl);
            vfloat64m1_t prod = __riscv_vfmul_vv_f64m1(av, xv, vl);
            vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m1_t red = __riscv_vfredusum_vs_f64m1_f64m1(prod, zero, vl);
            sum += __riscv_vfmv_f_s_f64m1_f64(red);
            j += (int)vl;
        }
        y[i] = sum;
    }
}
