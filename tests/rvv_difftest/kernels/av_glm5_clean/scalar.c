/* saxpy: out[i] = 2*a[i] + b[i]  (f32 elementwise, bit-tier=float ULP) */
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) out[i] = 2.0f * a[i] + b[i];
}
#include "harness.h"
