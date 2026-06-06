#include <riscv_vector.h>
void kern(int n, double a, const double *x, double *y){
  for(size_t vl; n>0; n-=vl, x+=vl, y+=vl){ vl=__riscv_vsetvl_e64m8(n);
    vfloat64m8_t vx=__riscv_vle64_v_f64m8(x,vl), vy=__riscv_vle64_v_f64m8(y,vl);
    vy=__riscv_vfmacc_vf_f64m8(vy,a,vx,vl); __riscv_vse64_v_f64m8(y,vy,vl); } }