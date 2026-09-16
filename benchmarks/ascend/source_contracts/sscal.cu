// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasSscal(handle,n,&alpha,x,1);
// Reference semantics (single precision):
//   for(i) x[i] = alpha*x[i];
void run_sscal(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasSscal(handle,n,&alpha,x,1);
}
