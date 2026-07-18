"""Batch-run AgentVec-generated L1 BLAS kernels through the real OpenBLAS
benchmarks. Reports correctness (pass/total) and avg speedup vs generic scalar."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
from blas_lower import emit_l1, L1Spec

DST = "/path/to/operator_extract"; TOOL = f"{DST}/toolchain"
GCC = f"{TOOL}/riscv-gcc-14.3/bin/riscv64-linux-gcc"; QEMU = f"{TOOL}/qemu-riscv64/qemu-riscv64"
OPB = f"{DST}/openblas"; LIB = f"{OPB}/_file/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a"
INC = f"{OPB}/_file/include"; BENCH = f"{OPB}/_file/benchmark"

OPS = [("daxpy","13","axpy"), ("dscal","24","scal"), ("dcopy","14","copy"),
       ("ddot","15","dot"), ("dasum","11","asum")]
RNG = "16 256 16"


def run_op(opname, opnum, kind):
    opdir = f"{OPB}/{opnum}_{opname}_op"; work = f"/path/to/agentvec_lab/runs/{opname}"
    ksrc = emit_l1(L1Spec(name=f"{opname}_k_rvv", dtype="f64", kind=kind))
    local = os.path.join(os.path.dirname(__file__), "_gen", f"{opname}_k_rvv.c")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    open(local, "w", newline="\n").write(ksrc)
    run("server", f"mkdir -p {work}")
    put("server", local, f"{work}/{opname}_k_rvv.c")
    build = (f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math "
             f"-w -DSHOW_TIME -DDOUBLE {BENCH}/_{opname}_dgc.c {opdir}/{opname}.c {opdir}/{opname}_k.c "
             f"{work}/{opname}_k_rvv.c {LIB} -I{INC} -DSMP_SERVER -static -lc -lm -lpthread "
             f"-o {opname}.out 2>&1 && echo BUILD_OK")
    rc, out, err = run("server", build, timeout=240)
    if "BUILD_OK" not in out:
        return {"op": opname, "build": "FAIL", "err": (out[-300:] + err[-200:])}
    rc, out, err = run("server", f"cd {work} && {QEMU} -cpu rv64,v=true,vlen=256,vext_spec=v1.0 "
                                  f"{opname}.out {RNG} 2>&1", timeout=180)
    results = re.findall(r"=>\s*(pass|fail)", out)
    passed = results.count("pass"); total = len(results)
    msp = re.search(r"Average Speedup:\s*([0-9.]+)", out)
    return {"op": opname, "build": "OK", "passed": passed, "total": total,
            "spr": round(passed/total, 3) if total else None,
            "avg_speedup": float(msp.group(1)) if msp else None}


if __name__ == "__main__":
    print(f"{'op':8} {'build':6} {'SPR':>8} {'pass/total':>12} {'avg_speedup':>12}")
    for opname, opnum, kind in OPS:
        r = run_op(opname, opnum, kind)
        if r["build"] != "OK":
            print(f"{r['op']:8} BUILD_FAIL  {r.get('err','')[:120]}")
        else:
            print(f"{r['op']:8} {r['build']:6} {str(r['spr']):>8} "
                  f"{str(r['passed'])+'/'+str(r['total']):>12} {str(r['avg_speedup']):>12}")
