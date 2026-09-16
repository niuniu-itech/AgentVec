// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasScopy(handle,n,x,1,y,1);
// Reference semantics (single precision):
//   for(i) y[i] = x[i];
void run_scopy(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasScopy(handle,n,x,1,y,1);
}
