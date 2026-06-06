/* Opus-4.8 pure-LLM gemv: m8 running accumulator + final reduce, VLA. */
#include <riscv_vector.h>
void gemv(int N, const double *A, const double *x, double *y){
  for(int i=0;i<N;i++){
    const double *row=A+(size_t)i*N; size_t vlmax=__riscv_vsetvlmax_e64m8();
    vfloat64m8_t acc=__riscv_vfmv_v_f_f64m8(0.0,vlmax);
    for(int j=0;j<N;){size_t vl=__riscv_vsetvl_e64m8((size_t)(N-j));
      vfloat64m8_t a=__riscv_vle64_v_f64m8(row+j,vl),xv=__riscv_vle64_v_f64m8(x+j,vl);
      acc=__riscv_vfmacc_vv_f64m8(acc,a,xv,vl);j+=(int)vl;}
    vfloat64m1_t z=__riscv_vfmv_v_f_f64m1(0.0,1);
    vfloat64m1_t s=__riscv_vfredusum_vs_f64m8_f64m1(acc,z,vlmax);
    y[i]=__riscv_vfmv_f_s_f64m1_f64(s);
  }
}
