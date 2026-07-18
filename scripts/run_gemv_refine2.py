"""Robust gemv LMUL refinement: interleaved rounds, median speedup, MS vs expert.
Denoises the board timing so the refinement curve is reliable."""
import os, sys, re, statistics; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get
from run_gemv_refine import gemv_kernel, GCC, OPB, LIB, INC, BENCH, opdir, work, RNG, LOOPS


def run_only(tag):
    bd = f"/home/user/agentvec_lab/gv_{tag}"
    o = run("board", f"cd {bd} && OPENBLAS_LOOPS={LOOPS} ./run.out {RNG} 2>&1 | grep 'Average Speedup'", timeout=300)[1]
    m = re.search(r"Average Speedup:\s*([0-9.]+)", o); return float(m.group(1)) if m else None


def build(kernel_remote, tag):
    b = (f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math -w -DSHOW_TIME -DDOUBLE "
         f"'{BENCH}/_dgemv_dgc.c' {opdir}/dgemv.c {opdir}/dgemv_k.c {kernel_remote} {LIB} -I{INC} -DSMP_SERVER -static -lc -lm -lpthread -o gv_{tag}.out 2>&1 && echo OK")
    if "OK" not in run("server", b, timeout=240)[1]: return False
    get("server", f"{work}/gv_{tag}.out", f"_board/gv_{tag}.out")
    bd = f"/home/user/agentvec_lab/gv_{tag}"; run("board", f"mkdir -p {bd}"); put("board", f"_board/gv_{tag}.out", f"{bd}/run.out"); run("board", f"chmod +x {bd}/run.out")
    return True


def main():
    run("server", f"mkdir -p {work}")
    configs = [("expert", f"{opdir}/dgemv_k_rvv.c")]
    for L in (1, 2, 4, 8):
        local = f"_gen/gemv_m{L}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(gemv_kernel(L))
        put("server", local, f"{work}/gemv_m{L}.c"); configs.append((f"m{L}", f"{work}/gemv_m{L}.c"))
    for tag, src in configs:
        if not build(src, tag): print(f"{tag}: BUILDFAIL", flush=True)
    samples = {tag: [] for tag, _ in configs}
    for rnd in range(3):
        for tag, _ in configs:
            sp = run_only(tag)
            if sp: samples[tag].append(sp)
        print(f"round {rnd+1}: " + "  ".join(f"{t}={statistics.median(samples[t]):.3f}" for t in samples if samples[t]), flush=True)
    med = {t: statistics.median(v) for t, v in samples.items() if v}
    exp = med["expert"]
    print(f"\n=== gemv MS vs expert (median of 3) ===  expert speedup={exp:.3f}")
    for L in (8, 4, 2, 1):
        t = f"m{L}"
        if t in med: print(f"  LMUL {t}: speedup={med[t]:.3f}  MS={med[t]/exp:.3f} ({100*med[t]/exp:.0f}% of expert)", flush=True)
    import json; json.dump({"median": med, "expert": exp}, open("_gemv_refine2.json", "w"))


if __name__ == "__main__":
    main()
