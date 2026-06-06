#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void kern(int n, double a, const double *x, double *y);
int main(void){int Ns[]={1,3,7,8,15,64,127,256,1000}; int fails=0; double mr=0;
 for(int t=0;t<9;t++){int n=Ns[t]; double a=1.7;
   double*x=malloc(8.0*n),*y=malloc(8.0*n),*r=malloc(8.0*n);
   for(int i=0;i<n;i++){x[i]=((i*2654435761u)%1000)/500.0-1.0; y[i]=((i*40503u+7u)%1000)/500.0-1.0; r[i]=y[i]+a*x[i];}
   kern(n,a,x,y);
   for(int i=0;i<n;i++){double d=fabs(y[i]-r[i])/(fabs(r[i])+1e-9); if(d>mr)mr=d;}
   if(mr>=1e-6)fails++; free(x);free(y);free(r);}
 printf("SUMMARY fails=%d/9 maxrel=%.1e\n",fails,mr); return 0;}