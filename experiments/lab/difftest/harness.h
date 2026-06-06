/* Common differential-testing driver.
 * A kernel .c file defines DT, DT_IS_FLOAT (0/1), REDUCE (0/1) and
 * agentvec_kernel(const DT*,const DT*,const DT*,DT*,int), then #includes this.
 * argv: seed n out_path. Inputs are generated deterministically from seed so
 * the scalar oracle build and the RVV build see identical inputs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n);

static uint32_t _st;
static uint32_t _xs(void) { _st ^= _st << 13; _st ^= _st >> 17; _st ^= _st << 5; return _st; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s seed n outpath\n", argv[0]); return 2; }
    uint32_t seed = (uint32_t)strtoul(argv[1], 0, 10);
    int n = atoi(argv[2]);
    const char *outp = argv[3];
    _st = seed ? seed : 1u;

    DT *a = malloc(sizeof(DT) * (size_t)n);
    DT *b = malloc(sizeof(DT) * (size_t)n);
    DT *c = malloc(sizeof(DT) * (size_t)n);
    int outlen = REDUCE ? 1 : n;
    DT *out = calloc((size_t)outlen, sizeof(DT));

    for (int i = 0; i < n; i++) {
#if DT_IS_FLOAT
        a[i] = (DT)((int32_t)(_xs() % 20001) - 10000) / (DT)100.0;
        b[i] = (DT)((int32_t)(_xs() % 20001) - 10000) / (DT)100.0;
        c[i] = (DT)((int32_t)(_xs() % 20001) - 10000) / (DT)100.0;
#else
        a[i] = (DT)((int32_t)(_xs() % 2001) - 1000);
        b[i] = (DT)((int32_t)(_xs() % 2001) - 1000);
        c[i] = (DT)((int32_t)(_xs() % 2001) - 1000);
#endif
    }

    agentvec_kernel(a, b, c, out, n);

    FILE *f = fopen(outp, "wb");
    if (!f) { perror("fopen"); return 3; }
    fwrite(out, sizeof(DT), (size_t)outlen, f);
    fclose(f);
    return 0;
}
