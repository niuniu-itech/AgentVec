#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)b;(void)c;float s=0;for(int i=0;i<n;i++)s+=a[i]*a[i];out[0]=sqrtf(s);}
#include "harness.h"
