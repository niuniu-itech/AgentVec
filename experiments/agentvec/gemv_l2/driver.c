#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void gemv(int N, const double *A, const double *x, double *y);   /* y = A x */
static void ref(int N,const double*A,const double*x,double*y){
  for(int i=0;i<N;i++){double s=0;for(int j=0;j<N;j++)s+=A[i*N+j]*x[j];y[i]=s;}}
int main(void){int Ns[]={1,2,3,5,7,8,15,16,31,33,64,127,128,255,256};int fails=0;double maxrel=0;
  for(int t=0;t<15;t++){int N=Ns[t];
    double*A=malloc(8.0*N*N),*x=malloc(8.0*N),*y=malloc(8.0*N),*r=malloc(8.0*N);
    for(int i=0;i<N*N;i++)A[i]=((i*1103515245u+12345u)%1000)/1000.0;
    for(int i=0;i<N;i++)x[i]=((i*16807u+7u)%1000)/1000.0;
    ref(N,A,x,r);for(int i=0;i<N;i++)y[i]=-999.0;gemv(N,A,x,y);
    double mr=0;for(int i=0;i<N;i++){double rel=fabs(y[i]-r[i])/(fabs(r[i])+1e-12);if(rel>mr)mr=rel;}
    int ok=mr<1e-6;if(!ok)fails++;if(mr>maxrel)maxrel=mr;
    free(A);free(x);free(y);free(r);}
  printf("SUMMARY fails=%d/15 maxrel=%.1e\n",fails,maxrel);return 0;}
