"""Slot an AgentVec-generated RVV kernel into a real OpenBLAS operator benchmark
and run it through the unmodified harness (correctness + speedup vs generic scalar).

Usage: python run_real_op.py daxpy 13 axpy f64
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
from blas_lower import emit_l1, L1Spec

DST = "/path/to/operator_extract"
TOOL = f"{DST}/toolchain"
GCC = f"{TOOL}/riscv-gcc-14.3/bin/riscv64-linux-gcc"
QEMU = f"{TOOL}/qemu-riscv64/qemu-riscv64"
OPB = f"{DST}/openblas"
LIB = f"{OPB}/_file/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a"
INC = f"{OPB}/_file/include"
BENCHDIR = f"{OPB}/_file/benchmark"


def main():
    opname = sys.argv[1] if len(sys.argv) > 1 else "daxpy"
    opnum = sys.argv[2] if len(sys.argv) > 2 else "13"
    kind = sys.argv[3] if len(sys.argv) > 3 else "axpy"
    dtype = sys.argv[4] if len(sys.argv) > 4 else "f64"
    rng = sys.argv[5] if len(sys.argv) > 5 else "16 200 16"

    opdir = f"{OPB}/{opnum}_{opname}_op"
    work = f"/path/to/agentvec_lab/runs/{opname}_agentvec"
    bench = f"{BENCHDIR}/_{opname}_dgc.c"
    define = "-DDOUBLE" if dtype == "f64" else "-DSINGLE"

    # 1) generate the AgentVec RVV kernel (deterministic lowering)
    ksrc = emit_l1(L1Spec(name=f"{opname}_k_rvv", dtype=dtype, kind=kind))
    local = os.path.join(os.path.dirname(__file__), "_gen", f"{opname}_k_rvv.c")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    open(local, "w", newline="\n").write(ksrc)

    run("server", f"mkdir -p {work}")
    put("server", local, f"{work}/{opname}_k_rvv.c")

    # 2) build: benchmark + interface + generic scalar kernel + AgentVec RVV kernel + openblas lib
    build = (f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math "
             f"-w -DSHOW_TIME {define} {bench} {opdir}/{opname}.c {opdir}/{opname}_k.c "
             f"{work}/{opname}_k_rvv.c {LIB} -I{INC} -DSMP_SERVER -static -lc -lm -lpthread "
             f"-o {opname}_agentvec.out 2>&1 && echo BUILD_OK")
    rc, out, err = run("server", build, timeout=240)
    if "BUILD_OK" not in out:
        print("BUILD FAILED:\n", out[-1500:], err[-500:]); return

    # 3) run through the unmodified harness
    rc, out, err = run("server", f"cd {work} && {QEMU} -cpu rv64,v=true,vlen=256,vext_spec=v1.0 "
                                  f"{opname}_agentvec.out {rng} 2>&1", timeout=180)
    print(out)


if __name__ == "__main__":
    main()
