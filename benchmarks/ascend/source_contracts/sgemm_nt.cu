// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasSgemm(handle,N,T,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc);
// Reference semantics (single precision):
//   for(i,j){ s=0; for(k) s+=A[i][k]*B[j][k]; C[i][j]=alpha*s+beta*C[i][j]; } // A*B^T
void run_sgemm_nt(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasSgemm(handle,N,T,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc);
}
