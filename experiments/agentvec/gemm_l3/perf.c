#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void gemm(int N,const double*A,const double*B,double*C);
int main(int argc,char**argv){int N=argc>1?atoi(argv[1]):512;
  double*A=malloc(8.0*N*N),*B=malloc(8.0*N*N),*C=malloc(8.0*N*N);
  for(int i=0;i<N*N;i++){A[i]=((i*2654435761u)%1000)/1000.0;B[i]=((i*40503u+7u)%1000)/1000.0;}
  gemm(N,A,B,C); /* warmup */
  struct timespec t0,t1;double best=1e9;
  for(int r=0;r<7;r++){clock_gettime(CLOCK_MONOTONIC,&t0);gemm(N,A,B,C);clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;if(dt<best)best=dt;}
  printf("N=%d GFLOPS=%.3f\n",N,2.0*N*N*N/best*1e-9);return 0;}
