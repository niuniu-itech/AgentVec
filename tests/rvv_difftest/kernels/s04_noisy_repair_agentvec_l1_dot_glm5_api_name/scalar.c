#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
#define OUT_FACTOR 1
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c;
    float s=0.0f; for(int i=0;i<n;i++) s += a[i]*b[i]; out[0]=s;
}
#include "harness.h"
