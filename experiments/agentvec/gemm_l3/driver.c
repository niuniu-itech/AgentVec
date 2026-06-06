#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
void gemm(int N, const double *A, const double *B, double *C);   /* from kernel.c */
static void ref(int N,const double*A,const double*B,double*C){
  for(int i=0;i<N;i++)for(int j=0;j<N;j++){double s=0;for(int k=0;k<N;k++)s+=A[i*N+k]*B[k*N+j];C[i*N+j]=s;}}
int main(void){
  int Ns[]={1,2,3,5,7,8,15,16,31,33,64,127,128,255,256}; int fails=0; double maxrel=0;
  for(int t=0;t<15;t++){int N=Ns[t];
    double*A=malloc(8.0*N*N),*B=malloc(8.0*N*N),*C=malloc(8.0*N*N),*R=malloc(8.0*N*N);
    for(int i=0;i<N*N;i++){A[i]=((i*1103515245u+12345u)%1000)/1000.0; B[i]=((i*16807u+7u)%1000)/1000.0;}
    ref(N,A,B,R); for(int i=0;i<N*N;i++)C[i]=-999.0; gemm(N,A,B,C);
    double mr=0; for(int i=0;i<N*N;i++){double rel=fabs(C[i]-R[i])/(fabs(R[i])+1e-12); if(rel>mr)mr=rel;}
    int ok=mr<1e-6; if(!ok)fails++; if(mr>maxrel)maxrel=mr;
    printf("  N=%4d maxrel=%.1e %s\n",N,mr,ok?"ok":"FAIL"); free(A);free(B);free(C);free(R);}
  int N=256; double*A=malloc(8.0*N*N),*B=malloc(8.0*N*N),*C=malloc(8.0*N*N);
  for(int i=0;i<N*N;i++){A[i]=0.5;B[i]=0.3;} struct timespec t0,t1; double best=1e9;
  for(int r=0;r<3;r++){clock_gettime(CLOCK_MONOTONIC,&t0);gemm(N,A,B,C);clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9; if(dt<best)best=dt;}
  printf("SUMMARY fails=%d/15 maxrel=%.1e gflops=%.3f\n",fails,maxrel,2.0*N*N*N/best*1e-9);
  return 0;}
