/* AgentVec deterministic gemv: per-row vectorized dot, VLA by construction. */
#include <riscv_vector.h>
void gemv(int N, const double *A, const double *x, double *y){
  size_t vlmax=__riscv_vsetvlmax_e64m8();
  for(int i=0;i<N;i++){const double*Ar=&A[(size_t)i*N];
    vfloat64m8_t acc=__riscv_vfmv_v_f_f64m8(0.0,vlmax);
    for(int j=0;j<N;){size_t vl=__riscv_vsetvl_e64m8((size_t)(N-j));
      vfloat64m8_t va=__riscv_vle64_v_f64m8(Ar+j,vl),vx=__riscv_vle64_v_f64m8(x+j,vl);
      acc=__riscv_vfmacc_vv_f64m8(acc,va,vx,vl);j+=(int)vl;}
    vfloat64m1_t r=__riscv_vfmv_v_f_f64m1(0.0,1);
    r=__riscv_vfredusum_vs_f64m8_f64m1(acc,r,vlmax);
    double res;__riscv_vse64_v_f64m1(&res,r,1);y[i]=res;}}
