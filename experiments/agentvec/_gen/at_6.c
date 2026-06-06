#include <riscv_vector.h>
#define BK 64
#define BJ 128
void gemm(int N,const double*A,const double*B,double*C){
  for(size_t t=0;t<(size_t)N*N;t++)C[t]=0.0;
  for(int jj=0;jj<N;jj+=BJ){ int jm=jj+BJ<N?jj+BJ:N;
    for(int kk=0;kk<N;kk+=BK){ int km=kk+BK<N?kk+BK:N;
      for(int i=0;i<N;i++){ const double*Ar=&A[(size_t)i*N]; double*Cr=&C[(size_t)i*N];
        for(int k=kk;k<km;k++){ double a=Ar[k]; const double*Br=&B[(size_t)k*N];
          for(int j=jj;j<jm;){ size_t vl=__riscv_vsetvl_e64m4((size_t)(jm-j));
            vfloat64m4_t vc=__riscv_vle64_v_f64m4(Cr+j,vl),vb=__riscv_vle64_v_f64m4(Br+j,vl);
            vc=__riscv_vfmacc_vf_f64m4(vc,a,vb,vl); __riscv_vse64_v_f64m4(Cr+j,vc,vl); j+=(int)vl; }}}}}}