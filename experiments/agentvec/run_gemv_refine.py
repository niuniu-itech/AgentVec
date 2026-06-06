"""Closed-loop refinement on L2 gemv: the deterministic column-axpy lowering has one
free parameter (LMUL); tune it via on-hardware feedback and watch MS vs the hand-written
OpenBLAS expert converge to parity. Unlike GEMM (L3), the expert's edge here is LMUL, not
packing, so the search space contains it."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get
DST = "/path/to/operator_extract"; TOOL = f"{DST}/toolchain"
GCC = f"{TOOL}/riscv-gcc-14.3/bin/riscv64-linux-gcc"; OPB = f"{DST}/openblas"
LIB = f"{OPB}/_file/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a"; INC = f"{OPB}/_file/include"; BENCH = f"{OPB}/_file/benchmark"
opdir = f"{OPB}/17_dgemv_op"; work = "/path/to/agentvec_lab/gemv_bench"; RNG = "1024 4096 1024"; LOOPS = "20"
HDR = '#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <math.h>\n#include <sys/time.h>\n#include <riscv_vector.h>\n#include "common.h"\n'


def gemv_kernel(L):
    return HDR + f"""
int dgemv_n_rvv(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha, double *a, BLASLONG lda, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *buffer) {{
    if(n < 0) return(0);
    double *a_ptr, *x_ptr; BLASLONG i; vfloat64m{L}_t va, vy;
    if(inc_y == 1) {{
        for (size_t vl; m > 0; m -= vl, y += vl, a += vl) {{
            vl = __riscv_vsetvl_e64m{L}(m); a_ptr = a; x_ptr = x;
            vy = __riscv_vle64_v_f64m{L}(y, vl);
            for(i = 0; i < n; i++) {{ va = __riscv_vle64_v_f64m{L}(a_ptr, vl);
                vy = __riscv_vfmacc_vf_f64m{L}(vy, (alpha * (*x_ptr)), va, vl); a_ptr += lda; x_ptr += inc_x; }}
            __riscv_vse64_v_f64m{L}(y, vy, vl);
        }}
    }} else {{
        BLASLONG stride_y = inc_y * sizeof(double);
        for (size_t vl; m > 0; m -= vl, y += vl*inc_y, a += vl) {{
            vl = __riscv_vsetvl_e64m{L}(m); a_ptr = a; x_ptr = x;
            vy = __riscv_vlse64_v_f64m{L}(y, stride_y, vl);
            for(i = 0; i < n; i++) {{ va = __riscv_vle64_v_f64m{L}(a_ptr, vl);
                vy = __riscv_vfmacc_vf_f64m{L}(vy, (alpha * (*x_ptr)), va, vl); a_ptr += lda; x_ptr += inc_x; }}
            __riscv_vsse64_v_f64m{L}(y, stride_y, vy, vl);
        }}
    }}
    return(0);
}}
"""


def build_run(kernel_remote, tag):
    b = (f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math -w -DSHOW_TIME -DDOUBLE "
         f"'{BENCH}/_dgemv_dgc.c' {opdir}/dgemv.c {opdir}/dgemv_k.c {kernel_remote} {LIB} -I{INC} -DSMP_SERVER -static -lc -lm -lpthread -o gv_{tag}.out 2>&1 && echo OK")
    if "OK" not in run("server", b, timeout=240)[1]: return None
    get("server", f"{work}/gv_{tag}.out", f"_board/gv_{tag}.out")
    bd = f"/home/user/agentvec_lab/gv_{tag}"; run("board", f"mkdir -p {bd}"); put("board", f"_board/gv_{tag}.out", f"{bd}/run.out"); run("board", f"chmod +x {bd}/run.out")
    o = run("board", f"cd {bd} && OPENBLAS_LOOPS={LOOPS} ./run.out {RNG} 2>&1 | grep 'Average Speedup'", timeout=300)[1]
    m = re.search(r"Average Speedup:\s*([0-9.]+)", o); return float(m.group(1)) if m else None


def main():
    run("server", f"mkdir -p {work}")
    exp = build_run(f"{opdir}/dgemv_k_rvv.c", "expert")
    print(f"expert dgemv avg speedup = {exp:.3f}", flush=True)
    rows = []; best = 0.0
    for it, L in enumerate([1, 2, 4, 8], 1):
        local = f"_gen/gemv_m{L}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(gemv_kernel(L))
        put("server", local, f"{work}/gemv_m{L}.c")
        sp = build_run(f"{work}/gemv_m{L}.c", f"m{L}")
        ms = sp / exp if (sp and exp) else 0.0; best = max(best, ms)
        rows.append((L, sp, ms, best))
        print(f"[{it}/4] LMUL m{L}: speedup={sp:.3f}  MS={ms:.3f} (best {best:.3f} = {100*best:.0f}% of expert)", flush=True)
    import json; json.dump({"expert_sp": exp, "rows": rows}, open("_gemv_refine.json", "w"))
    print(f"\nCONVERGED gemv best MS = {best:.3f} = {100*best:.0f}% of expert", flush=True)


if __name__ == "__main__":
    main()
