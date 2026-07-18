#include <math.h>
#define DT float
#define DT_IS_FLOAT 1
#define REDUCE 0
#define A_IS_MATRIX 0
#define B_IS_MATRIX 0
#define C_IS_MATRIX 1
#define OUT_IS_MATRIX 1
#define SYMMETRIC_A 0
#define LOWER_TRIANGULAR_A 0
#define BANDED_A_RADIUS -1
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)a; (void)b; (void)c;
    for(int i=0;i<n;i++){ for(int j=0;j<n;j++) out[i*n+j] = c[i*n+j] + 2.0f*a[i]*b[j]; }
}
#include "harness.h"
