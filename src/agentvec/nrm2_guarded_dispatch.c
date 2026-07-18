/* Guarded dispatch for the Euclidean norm (nrm2), self-consistent and auditable.
 *
 * Contract (matches paper Appendix B.1): the unit-stride RVV fast path is selected ONLY
 * when (a) inc_x == 1, (b) all inputs are finite (no NaN/Inf), and (c) max_i|x_i| is below
 * the overflow threshold thr = sqrt(0.25*DBL_MAX/n), which keeps sum_i x_i^2 finite.
 * Otherwise the dispatch falls back to the overflow-safe scaling reference (the original
 * idiom). Out-of-range inputs therefore change WHICH kernel runs, never the result.
 *
 * The earlier draft called the fast path with inc_x even though no strided fast path is
 * implemented; this version closes that gap conservatively by requiring inc_x == 1 for the
 * fast path (any inc_x != 1 falls back), so the code is consistent with what is implemented.
 *
 * Build (server): riscv-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d \
 *                 -static -lm nrm2_guarded_dispatch.c -o nrm2gd.out
 * Run  (board/qemu): qemu-riscv64 -cpu rv64,v=true,vlen=256,vext_spec=v1.0 ./nrm2gd.out
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <riscv_vector.h>

typedef long BLASLONG;

int g_path;  /* 0 = fast (RVV), 1 = fallback (scaling ref); set by dnrm2_dispatch for audit */

/* ---- unit-stride VLA fast path: sum of squares then sqrt (verbatim pipeline output) ---- */
double dnrm2_k_rvv(BLASLONG n, double *x, BLASLONG inc_x) {
    (void)inc_x;                                  /* fast path is unit-stride only */
    if (n <= 0) return 0.0;
    if (n == 1) return fabs(x[0]);
    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(0.0, vlmax), vx;
    for (size_t vl; n > 0; n -= vl, x += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vx = __riscv_vle64_v_f64m8(x, vl);
        acc = __riscv_vfmacc_vv_f64m8(acc, vx, vx, vl);
    }
    double res; vfloat64m1_t r = __riscv_vfmv_v_f_f64m1(0.0, 1);
    r = __riscv_vfredusum_vs_f64m8_f64m1(acc, r, vlmax);
    __riscv_vse64_v_f64m1(&res, r, 1);
    return sqrt(res);
}

/* ---- overflow-safe scaling reference (BLAS dnrm2 idiom; handles any magnitude/stride) ---- */
double dnrm2_ref(BLASLONG n, double *x, BLASLONG inc_x) {
    if (n <= 0) return 0.0;
    if (n == 1) return fabs(x[0]);
    double scale = 0.0, ssq = 1.0;
    for (BLASLONG i = 0; i < n; i++) {
        double xi = x[i * inc_x];
        if (xi != 0.0) {
            double a = fabs(xi);
            if (scale < a) { double t = scale / a; ssq = 1.0 + ssq * t * t; scale = a; }
            else           { double t = a / scale; ssq += t * t; }
        }
    }
    return scale * sqrt(ssq);
}

/* ---- guarded dispatch: O(n) range scan, fast path only when unit-stride+finite+bounded ---- */
double dnrm2_dispatch(BLASLONG n, double *x, BLASLONG inc_x) {
    if (n <= 1) { g_path = 1; return (n == 1) ? fabs(x[0]) : 0.0; }
    double m = 0.0; int finite = 1;
    for (BLASLONG i = 0; i < n; i++) {
        double a = fabs(x[i * inc_x]);
        if (!isfinite(a)) finite = 0;             /* NaN or +/-Inf anywhere -> fall back */
        else if (a > m) m = a;                    /* running max magnitude */
    }
    double thr = sqrt(0.25 * DBL_MAX / (double)n);
    int in_range = (inc_x == 1) && finite && (m <= thr);
    g_path = in_range ? 0 : 1;
    return in_range ? dnrm2_k_rvv(n, x, inc_x)    /* trace-licensed fast path */
                    : dnrm2_ref (n, x, inc_x);    /* overflow-safe scaling fallback */
}

/* --------------------------- negative-control test harness --------------------------- */
static int eq(double a, double b) {
    if (isnan(a) || isnan(b)) return isnan(a) && isnan(b);
    if (isinf(a) || isinf(b)) return a == b;
    double d = fabs(a - b), s = fabs(a) + fabs(b) + 1e-300;
    return d / s <= 1e-12;
}

int main(void) {
    BLASLONG n = 1000; int fails = 0;
    double *x = malloc((size_t)n * sizeof(double));
    printf("case            path   dispatch        reference       verdict\n");

    /* (0) in-range -> MUST take the fast path and match the reference */
    for (BLASLONG i = 0; i < n; i++) x[i] = sin(0.1 * i) * 3.0;
    double d = dnrm2_dispatch(n, x, 1), r = dnrm2_ref(n, x, 1);
    int ok = eq(d, r) && g_path == 0;
    printf("in-range        %-6s %.8e  %.8e  %s\n", g_path ? "FALL" : "FAST", d, r, ok ? "PASS" : "FAIL"); fails += !ok;

    /* (1) overflow (max|x|~1e160) -> fast path alone = +Inf; dispatch MUST fall back */
    for (BLASLONG i = 0; i < n; i++) x[i] = (i % 2 ? 1e160 : -1e160);
    d = dnrm2_dispatch(n, x, 1); r = dnrm2_ref(n, x, 1);
    double fastalone = dnrm2_k_rvv(n, x, 1);
    ok = eq(d, r) && g_path == 1 && isinf(fastalone);
    printf("overflow        %-6s %.8e  %.8e  %s (fast-alone=%.2e)\n", g_path ? "FALL" : "FAST", d, r, ok ? "PASS" : "FAIL", fastalone); fails += !ok;

    /* (2) NaN present -> caught by m==m; dispatch MUST fall back */
    for (BLASLONG i = 0; i < n; i++) x[i] = 1.0; x[7] = NAN;
    d = dnrm2_dispatch(n, x, 1); r = dnrm2_ref(n, x, 1);
    ok = eq(d, r) && g_path == 1;
    printf("NaN/Inf         %-6s %.8e  %.8e  %s\n", g_path ? "FALL" : "FAST", d, r, ok ? "PASS" : "FAIL"); fails += !ok;

    /* (3) strided inc_x=2 -> not unit-stride; dispatch MUST fall back and match ref */
    for (BLASLONG i = 0; i < n; i++) x[i] = cos(0.07 * i) * 2.0;
    d = dnrm2_dispatch(500, x, 2); r = dnrm2_ref(500, x, 2);
    ok = eq(d, r) && g_path == 1;
    printf("strided inc=2   %-6s %.8e  %.8e  %s\n", g_path ? "FALL" : "FAST", d, r, ok ? "PASS" : "FAIL"); fails += !ok;

    /* (4) boundary n<=1 */
    d = dnrm2_dispatch(1, x, 1); double d0 = dnrm2_dispatch(0, x, 1);
    ok = eq(d, fabs(x[0])) && d0 == 0.0;
    printf("boundary n<=1   %-6s n=1->%.8e n=0->%.1e            %s\n", "FALL", d, d0, ok ? "PASS" : "FAIL"); fails += !ok;

    free(x);
    printf("RESULT: %s (%d failure%s)\n", fails ? "FAIL" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
