#include <stdio.h>
void softmax(int n,const float*x,float*y);
int main(){int n=8; float x[8]={1,2,3,4,3,2,1,0},y[8]; printf("start\n"); fflush(stdout);
  softmax(n,x,y); printf("done: "); for(int i=0;i<n;i++)printf("%.4f ",y[i]); printf("\n"); return 0;}