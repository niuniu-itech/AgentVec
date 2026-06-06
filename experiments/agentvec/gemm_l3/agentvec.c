/* AgentVec deterministic lowering: naive VLA (no tiling), correct by construction. */
#include <riscv_vector.h>
void gemm(int N, const double *A, const double *B, double *C){
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++)C[i*N+j]=0.0;
    for(int k=0;k<N;k++){double a=A[i*N+k];const double*Br=&B[k*N];double*Cr=&C[i*N];
      for(int j=0;j<N;){size_t vl=__riscv_vsetvl_e64m1((size_t)(N-j));
        vfloat64m1_t vc=__riscv_vle64_v_f64m1(Cr+j,vl),vb=__riscv_vle64_v_f64m1(Br+j,vl);
        vc=__riscv_vfmacc_vf_f64m1(vc,a,vb,vl);__riscv_vse64_v_f64m1(Cr+j,vc,vl);j+=(int)vl;}}}}
