/* Cache-blocked GEMM: tile (j,k) so a B-block stays L1-resident and is reused
 * across all i rows; inner ikj order streams B sequentially. Target-aware
 * "ceiling" reference for what a cost-model-guided lowering would emit. */
#include <riscv_vector.h>
#define BK 64
#define BJ 64
void gemm(int N,const double*A,const double*B,double*C){
  for(int t=0;t<N*N;t++) C[t]=0.0;
  for(int jj=0;jj<N;jj+=BJ){ int jm=jj+BJ<N?jj+BJ:N;
    for(int kk=0;kk<N;kk+=BK){ int km=kk+BK<N?kk+BK:N;
      for(int i=0;i<N;i++){
        for(int k=kk;k<km;k++){ double a=A[(size_t)i*N+k]; const double*Br=&B[(size_t)k*N];
          for(int j=jj;j<jm;){ size_t vl=__riscv_vsetvl_e64m1((size_t)(jm-j));
            vfloat64m1_t vc=__riscv_vle64_v_f64m1(C+(size_t)i*N+j,vl);
            vfloat64m1_t vb=__riscv_vle64_v_f64m1(Br+j,vl);
            vc=__riscv_vfmacc_vf_f64m1(vc,a,vb,vl);
            __riscv_vse64_v_f64m1(C+(size_t)i*N+j,vc,vl); j+=(int)vl; }}}}}}
