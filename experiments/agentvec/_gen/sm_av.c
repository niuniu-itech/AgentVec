#include <riscv_vector.h>
#include <math.h>
void softmax(int n, const float *x, float *y){
  if(n<=0) return;
  size_t vlmax=__riscv_vsetvlmax_e32m8();
  /* stage 1: m = max_i x_i   (contract max) */
  vfloat32m8_t vm=__riscv_vfmv_v_f_f32m8(-3.4e38f,vlmax);
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i)); vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); vm=__riscv_vfmax_vv_f32m8(vm,v,vl); i+=(int)vl;}
  vfloat32m1_t rm=__riscv_vfmv_v_f_f32m1(-3.4e38f,1); rm=__riscv_vfredmax_vs_f32m8_f32m1(vm,rm,vlmax);
  float m=__riscv_vfmv_f_s_f32m1_f32(rm);
  /* stage 2: y_i = exp(x_i - m); s = sum y_i   (map exp ; contract +) */
  float s=0.0f; for(int i=0;i<n;i++){ float e=expf(x[i]-m); y[i]=e; s+=e; }
  /* stage 3: y_i *= 1/s   (map) */
  float inv=1.0f/s;
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i)); vfloat32m8_t v=__riscv_vle32_v_f32m8(y+i,vl); v=__riscv_vfmul_vf_f32m8(v,inv,vl); __riscv_vse32_v_f32m8(y+i,v,vl); i+=(int)vl;}
}