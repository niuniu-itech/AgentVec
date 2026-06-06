/* Leaky ReLU scalar oracle (semantics of the oneDNN CUDA kernel + migrated _k.c):
 * out[i] = x>0 ? x : x*slope. */
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c;
    const float slope = 0.01f;
    for (int i = 0; i < n; i++) { float x = a[i]; out[i] = x > 0.0f ? x : x * slope; }
}
#include "harness.h"
