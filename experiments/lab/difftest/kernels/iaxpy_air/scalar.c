/* iaxpy: out[i] = 3*a[i] - b[i]  (i32 elementwise; bit-exact tier) */
#include <stdint.h>
#define DT int32_t
#define DT_IS_FLOAT 0
#define REDUCE 0
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)c;
    for (int i = 0; i < n; i++) out[i] = 3 * a[i] - b[i];
}
#include "harness.h"
