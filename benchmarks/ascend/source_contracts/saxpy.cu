// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasSaxpy(handle,n,&alpha,x,1,y,1);
// Reference semantics (single precision):
//   for(i) y[i] = alpha*x[i] + y[i];
void run_saxpy(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasSaxpy(handle,n,&alpha,x,1,y,1);
}
