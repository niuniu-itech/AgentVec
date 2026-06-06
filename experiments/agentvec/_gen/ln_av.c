#include <riscv_vector.h>
#include <math.h>
void layernorm(int n, const float *x, float *y){
  if(n<=0) return; size_t vlmax=__riscv_vsetvlmax_e32m8();
  /* stage 1: s = sum x ; mean = s/n   (contract +) */
  vfloat32m8_t vs=__riscv_vfmv_v_f_f32m8(0.0f,vlmax);
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i)); vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); vs=__riscv_vfadd_vv_f32m8(vs,v,vl); i+=(int)vl;}
  vfloat32m1_t r0=__riscv_vfmv_v_f_f32m1(0.0f,1); r0=__riscv_vfredusum_vs_f32m8_f32m1(vs,r0,vlmax);
  float mean=__riscv_vfmv_f_s_f32m1_f32(r0)/(float)n;
  /* stage 2: var = (1/n) sum (x-mean)^2   (map sub,sq ; contract +) */
  vfloat32m8_t va=__riscv_vfmv_v_f_f32m8(0.0f,vlmax);
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i));
    vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); v=__riscv_vfsub_vf_f32m8(v,mean,vl);
    va=__riscv_vfmacc_vv_f32m8(va,v,v,vl); i+=(int)vl;}
  vfloat32m1_t r1=__riscv_vfmv_v_f_f32m1(0.0f,1); r1=__riscv_vfredusum_vs_f32m8_f32m1(va,r1,vlmax);
  float var=__riscv_vfmv_f_s_f32m1_f32(r1)/(float)n;
  float rstd=1.0f/sqrtf(var+1e-5f);
  /* stage 3: y = (x-mean)*rstd   (map) */
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i));
    vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); v=__riscv_vfsub_vf_f32m8(v,mean,vl); v=__riscv_vfmul_vf_f32m8(v,rstd,vl);
    __riscv_vse32_v_f32m8(y+i,v,vl); i+=(int)vl;}
}