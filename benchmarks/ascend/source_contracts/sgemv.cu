// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasSgemv(handle,N,m,n,&alpha,A,lda,x,1,&beta,y,1);
// Reference semantics (single precision):
//   for(i){ s=0; for(j) s+=A[i][j]*x[j]; y[i]=alpha*s+beta*y[i]; }
void run_sgemv(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasSgemv(handle,N,m,n,&alpha,A,lda,x,1,&beta,y,1);
}
