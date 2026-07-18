#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define OUT_FACTOR 2
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c;
    for(int i=0;i<n;i++){ out[i] = b[i]; out[n+i] = a[i]; }
}
#include "harness.h"
