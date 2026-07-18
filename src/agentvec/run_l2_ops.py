"""Extend the L2 study beyond gemv: measure AgentVec's deterministic lowering vs the
hand-written OpenBLAS expert RVV kernel on real K1 hardware, per operator.

MS = (AgentVec migration speedup vs scalar) / (expert migration speedup vs scalar);
MS ~ 1.0 means AgentVec's lowered kernel reaches expert parity. Each op's AgentVec kernel
is the deterministic lowering of the recovered intent (vectorize over the free index),
with LMUL as the single tuned knob (as in run_gemv_refine.py). Correctness is the
benchmark's own per-size pass/fail (differential vs the scalar reference).

dger / dsymv / dspmv are vectorizable (map / map+reduce). dtrsv has a loop-carried
dependence -> the Guard must REJECT it (handled separately in check_trsv_guard.py)."""
import os, sys, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get

DST = "/data/lyw/operator_extract"; TOOL = f"{DST}/toolchain"
GCC = f"{TOOL}/riscv-gcc-14.3/bin/riscv64-linux-gcc"; OPB = f"{DST}/openblas"
LIB = f"{OPB}/_file/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a"
INC = f"{OPB}/_file/include"; BENCH = f"{OPB}/_file/benchmark"
WORK = "/data/lyw/agentvec_lab/l2_bench"; BOARD = "/home/dgc/agentvec_lab"; LOOPS = "20"
HDR = ('#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <math.h>\n'
       '#include <sys/time.h>\n#include <riscv_vector.h>\n#include "common.h"\n')


# ---- AgentVec deterministic-lowering kernels (LMUL = L is the single tuned knob) ----
def ger_kernel(L):
    # rank-1 update A[:,j] += (alpha*y[j]) * x[:]  -> column-axpy, vectorized over rows i
    return HDR + f"""
int dger_k_rvv(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha,
  double *x, BLASLONG incx, double *y, BLASLONG incy,
  double *a, BLASLONG lda, double *buffer) {{
    double *X = x;
    if (incx != 1) {{ X = buffer; for (BLASLONG i=0;i<m;i++) X[i]=x[i*incx]; }}
    for (BLASLONG j=0; j<n; j++) {{
        double yj = alpha * y[j*incy];
        double *ap = a + j*lda, *xp = X; BLASLONG mm = m;
        for (size_t vl; mm>0; mm-=vl, ap+=vl, xp+=vl) {{
            vl = __riscv_vsetvl_e64m{L}(mm);
            vfloat64m{L}_t va = __riscv_vle64_v_f64m{L}(ap, vl);
            vfloat64m{L}_t vx = __riscv_vle64_v_f64m{L}(xp, vl);
            va = __riscv_vfmacc_vf_f64m{L}(va, yj, vx, vl);
            __riscv_vse64_v_f64m{L}(ap, va, vl);
        }}
    }}
    return 0;
}}
"""


def _colredbody(L, aptr_decl, advance):
    # shared per-column axpy+dot body: Y[i]+=alpha*(A[i,i]*X[i] + sum_{k>i}A[k,i]*X[k]); Y[k>i]+=alpha*X[i]*A[k,i]
    return f"""
        double xi = X[i];
        {aptr_decl}
        double diag = col[0];
        BLASLONG len = m - i - 1;
        double *ap = col + 1, *xp = X + i + 1, *yp = Y + i + 1;
        vfloat64m{L}_t vacc = __riscv_vfmv_v_f_f64m{L}(0.0, vlmax);
        BLASLONG k = 0;
        for (; k + (BLASLONG)vlmax <= len; k += vlmax) {{
            vfloat64m{L}_t va = __riscv_vle64_v_f64m{L}(ap+k, vlmax);
            vfloat64m{L}_t vx = __riscv_vle64_v_f64m{L}(xp+k, vlmax);
            vfloat64m{L}_t vy = __riscv_vle64_v_f64m{L}(yp+k, vlmax);
            vy = __riscv_vfmacc_vf_f64m{L}(vy, alpha*xi, va, vlmax);
            __riscv_vse64_v_f64m{L}(yp+k, vy, vlmax);
            vacc = __riscv_vfmacc_vv_f64m{L}(vacc, va, vx, vlmax);
        }}
        vfloat64m1_t vz = __riscv_vfmv_v_f_f64m1(0.0, 1);
        vfloat64m1_t vr = __riscv_vfredusum_vs_f64m{L}_f64m1(vacc, vz, vlmax);
        double dot = __riscv_vfmv_f_s_f64m1_f64(vr);
        for (; k < len; k++) {{ double aik = ap[k]; yp[k] += alpha*xi*aik; dot += aik*xp[k]; }}
        Y[i] += alpha*(diag*xi + dot);
        {advance}"""


def symv_kernel(L):
    # symmetric MV (lower, full storage): column i lives at &A[i,i], stride 1 down the column
    return HDR + f"""
int dsymv_L_rvv(BLASLONG m, BLASLONG offset, double alpha, double *a, BLASLONG lda,
   double *x, BLASLONG incx, double *y, BLASLONG incy, double *buffer) {{
    (void)offset; double *X = x, *Y = y;
    if (incx != 1) {{ X = buffer;     for (BLASLONG i=0;i<m;i++) X[i]=x[i*incx]; }}
    if (incy != 1) {{ Y = buffer + m; for (BLASLONG i=0;i<m;i++) Y[i]=y[i*incy]; }}
    size_t vlmax = __riscv_vsetvlmax_e64m{L}();
    for (BLASLONG i = 0; i < m; i++) {{{_colredbody(L, "double *col = a + i + i*lda;", "")}
    }}
    if (incy != 1) {{ for (BLASLONG i=0;i<m;i++) y[i*incy]=Y[i]; }}
    return 0;
}}
"""


def spmv_kernel(L):
    # symmetric packed MV (lower): column i is m-i contiguous packed elems; pointer advances by m-i
    return HDR + f"""
int dspmv_L_rvv(BLASLONG m, double alpha, double *a, double *x, BLASLONG incx,
   double *y, BLASLONG incy, double *buffer) {{
    double *X = x, *Y = y, *col = a;
    if (incx != 1) {{ X = buffer;     for (BLASLONG i=0;i<m;i++) X[i]=x[i*incx]; }}
    if (incy != 1) {{ Y = buffer + m; for (BLASLONG i=0;i<m;i++) Y[i]=y[i*incy]; }}
    size_t vlmax = __riscv_vsetvlmax_e64m{L}();
    for (BLASLONG i = 0; i < m; i++) {{{_colredbody(L, "", "col += (m - i);")}
    }}
    if (incy != 1) {{ for (BLASLONG i=0;i<m;i++) y[i*incy]=Y[i]; }}
    return 0;
}}
"""


def gemv_kernel(L):
    # gemv (notrans): y += alpha*A*x via column-axpy, vectorized over rows (re-measured for a consistent L2 set)
    return HDR + f"""
int dgemv_n_rvv(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha, double *a, BLASLONG lda,
   double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *buffer) {{
    if(n < 0) return(0);
    double *a_ptr, *x_ptr; BLASLONG i; vfloat64m{L}_t va, vy;
    if(inc_y == 1) {{
        for (size_t vl; m > 0; m -= vl, y += vl, a += vl) {{
            vl = __riscv_vsetvl_e64m{L}(m); a_ptr = a; x_ptr = x;
            vy = __riscv_vle64_v_f64m{L}(y, vl);
            for(i = 0; i < n; i++) {{ va = __riscv_vle64_v_f64m{L}(a_ptr, vl);
                vy = __riscv_vfmacc_vf_f64m{L}(vy, (alpha*(*x_ptr)), va, vl); a_ptr += lda; x_ptr += inc_x; }}
            __riscv_vse64_v_f64m{L}(y, vy, vl);
        }}
    }} else {{
        BLASLONG sy = inc_y * sizeof(double);
        for (size_t vl; m > 0; m -= vl, y += vl*inc_y, a += vl) {{
            vl = __riscv_vsetvl_e64m{L}(m); a_ptr = a; x_ptr = x;
            vy = __riscv_vlse64_v_f64m{L}(y, sy, vl);
            for(i = 0; i < n; i++) {{ va = __riscv_vle64_v_f64m{L}(a_ptr, vl);
                vy = __riscv_vfmacc_vf_f64m{L}(vy, (alpha*(*x_ptr)), va, vl); a_ptr += lda; x_ptr += inc_x; }}
            __riscv_vsse64_v_f64m{L}(y, sy, vy, vl);
        }}
    }}
    return(0);
}}
"""


OPS = {
    "dgemv": dict(opdir=f"{OPB}/17_dgemv_op", bench=f"{BENCH}/_dgemv_dgc.c",
                  srcs=["dgemv.c", "dgemv_k.c"], expert="dgemv_k_rvv.c",
                  rng="512 2048 512", env="", gen=gemv_kernel),
    "dger": dict(opdir=f"{OPB}/18_dger_op", bench=f"{BENCH}/_dger_dgc.c",
                 srcs=["dger.c", "dger_k.c"], expert="dger_k_rvv.c",
                 rng="512 2048 512", env="", gen=ger_kernel),
    "dsymv": dict(opdir=f"{OPB}/30_dsymv_op", bench=f"{BENCH}/_dsymv_dgc.c",
                  srcs=["dsymv.c", "dsymv_k.c"], expert="dsymv_k_rvv.c",
                  rng="512 2048 512", env="OPENBLAS_UPLO=L ", gen=symv_kernel),
    "dspmv": dict(opdir=f"{OPB}/25_dspmv_op", bench=f"{BENCH}/_dspmv_dgc.c",
                  srcs=["dspmv.c", "dspmv_k.c"], expert="dspmv_k_rvv.c",
                  rng="512 2048 512", env="OPENBLAS_UPLO=L ", gen=spmv_kernel),
}


def build_run(op, kernel_remote, tag, verbose=False):
    c = OPS[op]
    srcs = " ".join(f"{c['opdir']}/{s}" for s in c["srcs"])
    b = (f"cd {WORK} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math "
         f"-w -DSHOW_TIME -DDOUBLE '{c['bench']}' {srcs} {kernel_remote} {LIB} -I{INC} "
         f"-DSMP_SERVER -static -lc -lm -lpthread -o {op}_{tag}.out 2>&1 && echo BUILD_OK")
    r = run("server", b, timeout=300)[1]
    if "BUILD_OK" not in r:
        print(f"  [BUILD FAIL {op}/{tag}]\n{r[-1200:]}"); return None, r
    os.makedirs("_board", exist_ok=True)
    get("server", f"{WORK}/{op}_{tag}.out", f"_board/{op}_{tag}.out")
    bd = f"{BOARD}/{op}_{tag}"; run("board", f"mkdir -p {bd}")
    put("board", f"_board/{op}_{tag}.out", f"{bd}/run.out"); run("board", f"chmod +x {bd}/run.out")
    o = run("board", f"cd {bd} && {c.get('env','')}OPENBLAS_LOOPS={LOOPS} ./run.out {c['rng']} 2>&1", timeout=360)[1]
    if verbose:
        print(f"  --- board output {op}/{tag} ---\n{o}\n  ---")
    m = re.search(r"Average Speedup:\s*([0-9.]+)", o, re.I)
    return (float(m.group(1)) if m else None), o


def measure(op):
    c = OPS[op]
    run("server", f"mkdir -p {WORK}")
    print(f"\n==== {op} ====", flush=True)
    exp, o0 = build_run(op, f"{c['opdir']}/{c['expert']}", "expert", verbose=True)
    print(f"  expert avg speedup = {exp}", flush=True)
    rows = []; best = 0.0; best_L = None
    for L in [1, 2, 4, 8]:
        local = f"_gen/{op}_m{L}.c"; os.makedirs("_gen", exist_ok=True)
        open(local, "w", newline="\n").write(c["gen"](L))
        put("server", local, f"{WORK}/{op}_m{L}.c")
        sp, o = build_run(op, f"{WORK}/{op}_m{L}.c", f"m{L}")
        passed = ("fail" not in o.lower()) and (sp is not None)
        ms = (sp / exp) if (sp and exp) else 0.0
        if passed and ms > best:
            best, best_L = ms, L
        rows.append(dict(L=L, speedup=sp, ms=ms, passed=passed))
        print(f"  LMUL m{L}: speedup={sp} MS={ms:.3f} passed={passed}", flush=True)
    res = dict(op=op, expert_sp=exp, best_ms=best, best_L=best_L, rows=rows)
    json.dump(res, open(f"_l2_{op}.json", "w"), indent=2)
    print(f"  => {op} best MS = {best:.3f} (LMUL m{best_L})", flush=True)
    return res


if __name__ == "__main__":
    ops = sys.argv[1:] or ["dger"]
    out = {op: measure(op) for op in ops}
    print("\n==== SUMMARY ====")
    for op, r in out.items():
        print(f"{op}: best MS = {r['best_ms']:.3f} (expert speedup {r['expert_sp']}, LMUL m{r['best_L']})")
