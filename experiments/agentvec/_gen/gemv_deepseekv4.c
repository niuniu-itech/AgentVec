#include <riscv_vector.h>

void gemv(int N, const double *A, const double *x, double *y) {
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        const double *row = A + i * N;
        int vl;
        for (int j = 0; j < N; j += vl) {
            vl = __riscv_vsetvl_e64m1(N - j);
            vfloat64m1_t vec_x = __riscv_vle64_v_f64m1(&x[j], vl);
            vfloat64m1_t vec_a = __riscv_vle64_v_f64m1(&row[j], vl);
            vfloat64m1_t vec_prod = __riscv_vfmul_vv_f64m1(vec_a, vec_x, vl);
            vfloat64m1_t vec_sum = __riscv_vfmv_v_f_f64m1(sum, vl);
            vec_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_sum, vec_prod, vec_sum, vl);
            sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
        }
        y[i] = sum;
    }
}