#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)c;for(int i=0;i<n;i++)out[i]=a[i];}
#include "harness.h"
