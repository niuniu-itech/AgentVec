#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 1
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)c;
    (void)b; float scale=0.0f, ssq=1.0f;
    for (int i=0;i<n;i++){ float x=a[i]; if (x!=0.0f){ float ax=fabsf(x);
        if (scale<ax){ float r=scale/ax; ssq=1.0f+ssq*r*r; scale=ax; }
        else { float r=ax/scale; ssq+=r*r; } } }
    out[0] = (scale==0.0f)?0.0f: scale*sqrtf(ssq);
}
#include "harness.h"
