#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 1
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c;
    for(int i=0;i<n;i++) out[i] = 2.0f*a[i] + 3.0f*b[i];
}
#include "harness.h"
