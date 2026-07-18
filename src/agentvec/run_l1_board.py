"""Board MS for many L1 operators: build each OpenBLAS benchmark with the AgentVec
kernel and with the expert kernel, run both natively on the K1, MS = AV/expert speedup."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get, HOSTS
from blas_lower import emit_l1, L1Spec
DST = "/path/to/operator_extract"; TOOL = f"{DST}/toolchain"
GCC = f"{TOOL}/riscv-gcc-14.3/bin/riscv64-linux-gcc"; OPB = f"{DST}/openblas"
LIB = f"{OPB}/_file/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a"; INC = f"{OPB}/_file/include"; BENCH = f"{OPB}/_file/benchmark"
RNG = "4096 32768 4096"; LOOPS = "30"
OPS = [("daxpy","13","axpy"),("dscal","24","scal"),("dcopy","14","copy"),("ddot","15","dot"),
       ("dasum","11","asum"),("daxpby","12","axpby"),("dswap","28","swap"),("dmax","20","max"),("dmin","21","min")]


def build_run(op, num, kernel_path, tag):
    opdir = f"{OPB}/{num}_{op}_op"; work = f"/path/to/agentvec_lab/l1board/{op}_{tag}"
    b = (f"mkdir -p {work} && cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math "
         f"-w -DSHOW_TIME -DDOUBLE '{BENCH}/_{op}_dgc.c' {opdir}/{op}.c {opdir}/{op}_k.c {kernel_path} "
         f"{LIB} -I{INC} -DSMP_SERVER -static -lc -lm -lpthread -o {op}.out 2>&1 && echo OK")
    r = run("server", b, timeout=240)[1]
    if "OK" not in r:
        return None, r[-120:]
    lp = f"_board/l1_{op}_{tag}.out"; get("server", f"{work}/{op}.out", lp)
    bd = f"/home/user/agentvec_lab/l1_{op}_{tag}"; run("board", f"mkdir -p {bd}"); put("board", lp, f"{bd}/run.out")
    o = run("board", f"cd {bd} && chmod +x run.out && OPENBLAS_LOOPS={LOOPS} ./run.out {RNG} 2>&1", timeout=300)[1]
    res = re.findall(r"=>\s*(pass|fail)", o); sp = re.search(r"Average Speedup:\s*([0-9.]+)", o)
    return (float(sp.group(1)) if sp else None, res.count("pass"), len(res)), None


def main():
    print(f"{'op':9}{'AV_sp':>8}{'exp_sp':>8}{'MS':>7}{'AV pass':>9}", flush=True)
    out_rows = []
    for op, num, kind in OPS:
        # AgentVec kernel
        ksrc = emit_l1(L1Spec(name=f"{op}_k_rvv", dtype="f64", kind=kind))
        local = f"_gen/{op}_k_rvv.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(ksrc)
        work = f"/path/to/agentvec_lab/l1board/{op}_av"; run("server", f"mkdir -p {work}"); put("server", local, f"{work}/{op}_k_rvv.c")
        av, e1 = build_run(op, num, f"{work}/{op}_k_rvv.c", "av")
        ex, e2 = build_run(op, num, f"{OPB}/{num}_{op}_op/{op}_k_rvv.c", "exp")
        if av is None or ex is None:
            print(f"{op:9}  BUILD_FAIL {e1 or ''}{e2 or ''}", flush=True); continue
        ms = av[0]/ex[0] if (av[0] and ex[0]) else None
        print(f"{op:9}{av[0]:>8.3f}{ex[0]:>8.3f}{(ms if ms else 0):>7.3f}{str(av[1])+'/'+str(av[2]):>9}", flush=True)
        out_rows.append((op, av[0], ex[0], ms, av[1], av[2]))
    import json; json.dump(out_rows, open("_l1board.json", "w"))


if __name__ == "__main__":
    main()
