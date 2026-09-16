// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasSasum(handle,n,x,1,&res);
// Reference semantics (single precision):
//   res = 0; for(i) res += fabsf(x[i]);
void run_sasum(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasSasum(handle,n,x,1,&res);
}
