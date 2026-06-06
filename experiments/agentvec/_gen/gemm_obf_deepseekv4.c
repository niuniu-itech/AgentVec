#include <riscv_vector.h>

void gemm(int N, const double *A, const double *B, double *C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      int k = 0;
      while (k < N) {
        int vl = __riscv_vsetvl_e64m1(N - k);
        vfloat64m1_t vec_a = __riscv_vle64_v_f64m1(&A[i * N + k], vl);
        vfloat64m1_t vec_b = __riscv_vle64_v_f64m1(&B[k * N + j], vl);
        vfloat64m1_t vec_prod = __riscv_vfmul_vv_f64m1(vec_a, vec_b, vl);
        vfloat64m1_t vec_sum = __riscv_vfmv_v_f_f64m1(0.0, vl);
        vec_sum = __riscv_vfredusum_vs_f64m1_f64m1(vec_sum, vec_prod, vec_sum, vl);
        sum += __riscv_vfmv_f_s_f64m1_f64(vec_sum);
        k += vl;
      }
      C[i * N + j] = sum;
    }
  }
}