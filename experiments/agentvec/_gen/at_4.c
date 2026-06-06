#include <riscv_vector.h>
void gemm(int N,const double*A,const double*B,double*C){
  for(int i=0;i<N;i++){ for(int j=0;j<N;j++)C[i*N+j]=0.0;
    for(int k=0;k<N;k++){ double a=A[(size_t)i*N+k]; const double*Br=&B[(size_t)k*N]; double*Cr=&C[(size_t)i*N];
      for(int j=0;j<N;){ size_t vl=__riscv_vsetvl_e64m8((size_t)(N-j));
        vfloat64m8_t vc=__riscv_vle64_v_f64m8(Cr+j,vl),vb=__riscv_vle64_v_f64m8(Br+j,vl);
        vc=__riscv_vfmacc_vf_f64m8(vc,a,vb,vl); __riscv_vse64_v_f64m8(Cr+j,vc,vl); j+=(int)vl; }}}}