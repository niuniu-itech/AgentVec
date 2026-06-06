/* Opus-4.8 pure-LLM GEMM: 4-row register-tiled, VLA via vsetvl. */
#include <riscv_vector.h>
void gemm(int N, const double *A, const double *B, double *C){
  for(int i=0;i<N;i+=4){
    int r = (N-i<4)?(N-i):4;
    for(int j=0;j<N;){
      size_t vl=__riscv_vsetvl_e64m1((size_t)(N-j));
      vfloat64m1_t c0=__riscv_vfmv_v_f_f64m1(0.0,vl),c1=c0,c2=c0,c3=c0;
      for(int k=0;k<N;k++){
        vfloat64m1_t b=__riscv_vle64_v_f64m1(B+(size_t)k*N+j,vl);
        c0=__riscv_vfmacc_vf_f64m1(c0,A[(size_t)(i+0)*N+k],b,vl);
        if(r>1)c1=__riscv_vfmacc_vf_f64m1(c1,A[(size_t)(i+1)*N+k],b,vl);
        if(r>2)c2=__riscv_vfmacc_vf_f64m1(c2,A[(size_t)(i+2)*N+k],b,vl);
        if(r>3)c3=__riscv_vfmacc_vf_f64m1(c3,A[(size_t)(i+3)*N+k],b,vl);
      }
      __riscv_vse64_v_f64m1(C+(size_t)(i+0)*N+j,c0,vl);
      if(r>1)__riscv_vse64_v_f64m1(C+(size_t)(i+1)*N+j,c1,vl);
      if(r>2)__riscv_vse64_v_f64m1(C+(size_t)(i+2)*N+j,c2,vl);
      if(r>3)__riscv_vse64_v_f64m1(C+(size_t)(i+3)*N+j,c3,vl);
      j+=(int)vl;
    }
  }
}
