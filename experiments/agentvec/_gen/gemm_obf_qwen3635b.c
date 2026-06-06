#include <riscv_vector.h>

void gemm(int N, const double *A, const double *B, double *C) {
    // A is MxK, B is KxN, C is MxN
    // Here N is the dimension for all three (square matrices)
    // C[i][j] = sum_k A[i][k] * B[k][j]
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}