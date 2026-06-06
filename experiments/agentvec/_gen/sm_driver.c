#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void softmax(int n, const float *x, float *y);
static void ref(int n, const float *x, float *y){
  float m=-3.4e38f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
  float s=0; for(int i=0;i<n;i++){y[i]=expf(x[i]-m); s+=y[i];}
  for(int i=0;i<n;i++) y[i]/=s;
}
int main(void){int Ns[]={7,16,33,64}; int fails=0; double mr=0;
  for(int t=0;t<4;t++){int n=Ns[t];
    float*x=malloc(4.0*n),*y=malloc(4.0*n),*r=malloc(4.0*n);
    for(int i=0;i<n;i++) x[i]=((i*2654435761u)%1000)/100.0f-5.0f;
    ref(n,x,r); softmax(n,x,y);
    double sm=0; for(int i=0;i<n;i++){double d=fabs((double)y[i]-r[i])/(fabs(r[i])+1e-9); if(d>sm)sm=d;}
    if(sm>=1e-4)fails++; if(sm>mr)mr=sm; free(x);free(y);free(r);}
  printf("SUMMARY fails=%d/4 maxrel=%.1e\n",fails,mr); return 0;}