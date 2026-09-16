// cuBLAS source operator being migrated to Ascend310P1 (AscendC).
// INPUT to the intent-driven pipeline: intent.recover() recovers the AIR from this.
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS call:
//   cublasSnrm2(handle,n,x,1,&res);
// Reference semantics (single precision):
//   res = 0; for(i) res += x[i]*x[i]; res = sqrtf(res);
void run_snrm2(cublasHandle_t h, int n, const float* x, float* y, float alpha) {
    // cublasSnrm2(handle,n,x,1,&res);
}
