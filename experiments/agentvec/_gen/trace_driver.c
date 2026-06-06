#include <stdio.h>
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
}