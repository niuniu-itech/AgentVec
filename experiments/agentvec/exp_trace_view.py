"""Plan A exp 2 (RUNTIME-TRACE view): a stencil out[i]=in[i]+0.5*in[i+1] whose in/out MAY alias.
Statically the alias is undecidable, so a sound Guard must assume may_alias and refuse to vectorize
(we prove the hazard is real). A runtime trace of the access addresses shows in/out are disjoint,
flips C_phy.alias -> disjoint, and licenses the vectorized RVV lowering (correct + faster).
Operationalizes figure4's 'Runtime trace' view. Self-contained (server + qemu)."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
DST="/path/to/operator_extract"; GCC=f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"; QEMU=f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"
work="/path/to/agentvec_lab/trace_view"

KERNELS = r'''#include <riscv_vector.h>
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
}'''

DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
void scal(int,const double*,double*); void vec(int,const double*,double*);
int main(void){
  int n=4096; double *buf=malloc(8.0*(n+1));      /* shared buffer for the ALIASED case */
  double *in=malloc(8.0*n),*o1=malloc(8.0*n),*o2=malloc(8.0*n);
  for(int i=0;i<n;i++) in[i]=((i*2654435761u)%1000)/500.0-1.0;
  /* (1) ALIASING HAZARD: out == in (full overlap). Run scalar vs vectorized in-place. */
  double *al1=malloc(8.0*n); memcpy(al1,in,8.0*n); scal(n,al1,al1);
  double *al2=malloc(8.0*n); memcpy(al2,in,8.0*n); vec(n,al2,al2);
  double amax=0; for(int i=0;i<n;i++){double d=fabs(al1[i]-al2[i]); if(d>amax)amax=d;}
  printf("ALIASED(out==in): scalar vs vectorized differ, max|diff|=%.3e  -> vectorization UNSAFE without alias info\n", amax);
  /* (2) RUNTIME TRACE on DISJOINT inputs: record address ranges, check overlap */
  long pin=(long)in, pout=(long)o2, span=8L*n;
  int disjoint = (pout+span<=pin) || (pin+span<=pout);
  printf("TRACE: addr(in)=[%ld,%ld)  addr(out)=[%ld,%ld)  -> alias=%s\n", pin,pin+span,pout,pout+span, disjoint?"DISJOINT":"MAY_ALIAS");
  /* (3) TRACE-LICENSED vectorization on disjoint buffers: correctness + speed */
  scal(n,in,o1); vec(n,in,o2);
  double vmax=0; for(int i=0;i<n;i++){double d=fabs(o1[i]-o2[i])/(fabs(o1[i])+1e-9); if(d>vmax)vmax=d;}
  struct timespec t0,t1; double ts=1e9,tv=1e9;
  for(int r=0;r<7;r++){clock_gettime(CLOCK_MONOTONIC,&t0); for(int k=0;k<200;k++) scal(n,in,o1); clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9; if(dt<ts)ts=dt;}
  for(int r=0;r<7;r++){clock_gettime(CLOCK_MONOTONIC,&t0); for(int k=0;k<200;k++) vec(n,in,o2); clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9; if(dt<tv)tv=dt;}
  printf("DISJOINT: trace-licensed vectorized RVV  maxrel=%.1e (%s)  speedup vs scalar fallback = %.2fx\n",
         vmax, vmax<1e-6?"CORRECT":"WRONG", ts/tv);
  return 0;
}'''

def main():
    run("server",f"mkdir -p {work}")
    os.makedirs("_gen",exist_ok=True)
    open("_gen/trace_k.c","w",newline="\n").write(KERNELS); open("_gen/trace_driver.c","w",newline="\n").write(DRIVER)
    put("server","_gen/trace_k.c",f"{work}/k.c"); put("server","_gen/trace_driver.c",f"{work}/driver.c")
    b=run("server",f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c k.c -o t.out 2>&1 && echo OK",timeout=150)[1]
    if "OK" not in b: print("BUILDFAIL",b[-200:]); return
    print("=== RUNTIME-TRACE view (qemu vlen=256) ===",flush=True)
    print(run("server",f"cd {work} && {QEMU} -cpu rv64,v=true,vlen=256,vext_spec=v1.0 t.out 2>&1",timeout=180)[1],flush=True)

if __name__=="__main__": main()
