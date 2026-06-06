#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void layernorm(int n, const float *x, float *y);
static void ref(int n, const float *x, float *y){
  float s=0; for(int i=0;i<n;i++)s+=x[i]; float mean=s/n;
  float v=0; for(int i=0;i<n;i++){float d=x[i]-mean; v+=d*d;} float var=v/n;
  float rstd=1.0f/sqrtf(var+1e-5f);
  for(int i=0;i<n;i++) y[i]=(x[i]-mean)*rstd;
}
int main(void){int Ns[]={7,16,33,64,255,1000}; int fails=0; double mr=0;
  for(int t=0;t<6;t++){int n=Ns[t];
    float*x=malloc(4.0*n),*y=malloc(4.0*n),*r=malloc(4.0*n);
    for(int i=0;i<n;i++) x[i]=((i*2654435761u)%1000)/100.0f-5.0f;
    ref(n,x,r); layernorm(n,x,y);
    double sm=0; for(int i=0;i<n;i++){double d=fabs((double)y[i]-r[i])/(fabs(r[i])+1e-6); if(d>sm)sm=d;}
    if(sm>=1e-3)fails++; if(sm>mr)mr=sm; free(x);free(y);free(r);}
  printf("SUMMARY fails=%d/6 maxrel=%.1e\n",fails,mr); return 0;}