#include <riscv_vector.h>
#include <stddef.h>
/* scalar oracle (sequential): backward 2-point stencil -> write to out[i] feeds read at i+1 when aliased */
void scal(int n, const double *in, double *out){
  if(n>0) out[0]=in[0];
  for(int i=1;i<n;i++) out[i]=in[i]+0.5*in[i-1];
}
/* AgentVec vectorized lowering (valid ONLY when in/out disjoint -- the trace-licensed schedule) */
void vec(int n, const double *in, double *out){
  if(n>0) out[0]=in[0];
  int i=1; for(size_t vl; i<n; i+=vl){ vl=__riscv_vsetvl_e64m8((size_t)(n-i));
    vfloat64m8_t a=__riscv_vle64_v_f64m8(in+i,vl), b=__riscv_vle64_v_f64m8(in+i-1,vl);
    vfloat64m8_t r=__riscv_vfmacc_vf_f64m8(a,0.5,b,vl); __riscv_vse64_v_f64m8(out+i,r,vl); }
}