"""Build + verify (QEMU, multi-VLEN) + benchmark (real K1 board) the improved
register-tiled + cache-blocked + packed L3 GEMM. Sweeps the microkernel/blocking
config and keeps the best-so-far vs the OpenBLAS expert. [i/N] progress prints."""
import os, sys, re, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get
import gemm_opt as G

WORK, BDIR, GCC, QEMU, MARCH = G.WORK, G.BDIR, G.GCC, G.QEMU, G.MARCH


def build(tag, code):
    os.makedirs("_gen", exist_ok=True)
    local = f"_gen/{tag}.c"
    open(local, "w", newline="\n").write(code)
    put("server", local, f"{WORK}/{tag}.c")
    flags = f"-O3 -march={MARCH} -mabi=lp64d -static"
    cc = run("server",
             f"cd {WORK} && {GCC} {flags} driver.c {tag}.c -o {tag}_c.out 2>&1 && "
             f"{GCC} {flags} perf.c {tag}.c -o {tag}_p.out 2>&1 && echo BUILD_OK",
             timeout=240)[1]
    return "BUILD_OK" in cc, cc


def verify(tag, vlens=(256, 128)):
    for v in vlens:
        o = run("server",
                f"cd {WORK} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 {tag}_c.out 2>&1 | grep SUMMARY",
                timeout=240)[1]
        if not re.search(r"fails=0/15", o):
            return False, f"vlen{v}:{o.strip()[:60]}"
    return True, "ok"


def bench(tag, Ns=(512,), reps=3):
    get("server", f"{WORK}/{tag}_p.out", f"_board/{tag}_p.out")
    run("board", f"mkdir -p {BDIR}")
    put("board", f"_board/{tag}_p.out", f"{BDIR}/run.out")
    run("board", f"chmod +x {BDIR}/run.out")
    res = {}
    for N in Ns:
        best = 0.0
        for _ in range(reps):
            o = run("board", f"cd {BDIR} && ./run.out {N} 2>&1", timeout=300)[1]
            m = re.search(r"GFLOPS=([0-9.]+)", o)
            if m:
                best = max(best, float(m.group(1)))
        res[N] = best
    return res


def build_expert():
    """Build the OpenBLAS dgemm benchmark (authoritative expert) on the server."""
    OPB = "/data/lyw/operator_extract/openblas/_file"
    LIB = f"{OPB}/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a"
    INC = f"{OPB}/include"
    drv = r'''#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
int main(int argc,char**argv){int N=argc>1?atoi(argv[1]):512;
  double*A=malloc(8.0*N*N),*B=malloc(8.0*N*N),*C=malloc(8.0*N*N);
  for(int i=0;i<N*N;i++){A[i]=((i*2654435761u)%1000)/1000.0;B[i]=((i*40503u+7u)%1000)/1000.0;C[i]=0;}
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,C,N);
  struct timespec t0,t1;double best=1e9;
  for(int r=0;r<7;r++){clock_gettime(CLOCK_MONOTONIC,&t0);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,C,N);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;if(dt<best)best=dt;}
  printf("N=%d GFLOPS=%.3f\n",N,2.0*N*N*N/best*1e-9);return 0;}
'''
    os.makedirs("_gen", exist_ok=True)
    open("_gen/expert_dgemm.c", "w", newline="\n").write(drv)
    put("server", "_gen/expert_dgemm.c", f"{WORK}/expert_dgemm.c")
    cc = run("server",
             f"cd {WORK} && {GCC} -O3 -march={MARCH} -mabi=lp64d -static -I{INC} "
             f"expert_dgemm.c {LIB} -lm -o expert_dgemm.out 2>&1 && echo BUILD_OK",
             timeout=240)[1]
    if "BUILD_OK" not in cc:
        print("  [expert] build failed:", cc.strip()[:200], flush=True)
        return None
    return True  # built; caller benches with desired threading


def expert_st(Ns, reps=3):
    if not build_expert():
        return {}
    return bench_existing("expert_dgemm", Ns, reps, env="OPENBLAS_NUM_THREADS=1")


def bench_existing(server_bin, Ns, reps, env=""):
    get("server", f"{WORK}/{server_bin}.out", f"_board/{server_bin}.out")
    run("board", f"mkdir -p {BDIR}")
    put("board", f"_board/{server_bin}.out", f"{BDIR}/run.out")
    run("board", f"chmod +x {BDIR}/run.out")
    res = {}
    for N in Ns:
        best = 0.0
        for _ in range(reps):
            o = run("board", f"cd {BDIR} && {env} ./run.out {N} 2>&1", timeout=300)[1]
            m = re.search(r"GFLOPS=([0-9.]+)", o)
            if m:
                best = max(best, float(m.group(1)))
        res[N] = best
    return res


# ---- candidate configs (name -> kwargs to gen_kernel) ----
# Register budget: MR*L (+L for B) <= 32 physical vregs.
# This in-order core rewards higher LMUL (fewer instrs); register tiling removes
# C load/store traffic. Sweep LMUL x MR x packing.
CONFIGS = [
    # LMUL=2 (vl=8), various row tiles
    ("rt8_m2_pack",   dict(MR=8,  L=2, MC=128, KC=256, NC=256, pack=True)),
    ("rt12_m2_pack",  dict(MR=12, L=2, MC=120, KC=256, NC=256, pack=True)),
    ("rt14_m2_pack",  dict(MR=14, L=2, MC=126, KC=256, NC=256, pack=True)),
    # LMUL=4 (vl=16), smaller row tiles (MR*4+4<=32 -> MR<=7)
    ("rt4_m4_pack",   dict(MR=4,  L=4, MC=128, KC=256, NC=256, pack=True)),
    ("rt6_m4_pack",   dict(MR=6,  L=4, MC=126, KC=256, NC=256, pack=True)),
    ("rt7_m4_pack",   dict(MR=7,  L=4, MC=126, KC=256, NC=256, pack=True)),
    # packing on/off ablation at the best-guess shape
    ("rt6_m4_nopack", dict(MR=6,  L=4, MC=126, KC=256, NC=256, pack=False)),
    ("rt8_m2_nopack", dict(MR=8,  L=2, MC=128, KC=256, NC=256, pack=False)),
    # smaller blocks (fit L1/L2 tighter)
    ("rt6_m4_smallblk", dict(MR=6, L=4, MC=96, KC=128, NC=128, pack=True)),
]

# Round 2: refine around the rt6_m4_pack optimum (block-size sweep + MR=5).
CONFIGS_R2 = [
    ("r2_mr5_m4",      dict(MR=5,  L=4, MC=120, KC=256, NC=256, pack=True)),
    ("r2_mr6_nc128",   dict(MR=6,  L=4, MC=126, KC=256, NC=128, pack=True)),
    ("r2_mr6_nc192",   dict(MR=6,  L=4, MC=126, KC=256, NC=192, pack=True)),
    ("r2_mr6_kc128",   dict(MR=6,  L=4, MC=126, KC=128, NC=256, pack=True)),
    ("r2_mr6_kc192",   dict(MR=6,  L=4, MC=126, KC=192, NC=256, pack=True)),
    ("r2_mr6_kc512",   dict(MR=6,  L=4, MC=126, KC=512, NC=256, pack=True)),
    ("r2_mr6_mc60",    dict(MR=6,  L=4, MC=60,  KC=256, NC=256, pack=True)),
    ("r2_mr6_mc252",   dict(MR=6,  L=4, MC=252, KC=256, NC=256, pack=True)),
    ("r2_mr6_big",     dict(MR=6,  L=4, MC=252, KC=512, NC=512, pack=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", action="store_true", help="(re)measure OpenBLAS expert")
    ap.add_argument("--expertval", type=float, default=None, help="use this expert GFLOP/s")
    ap.add_argument("--only", default=None, help="run only configs whose name contains this")
    ap.add_argument("--round", default="1", help="config set: 1 or 2")
    ap.add_argument("--scaling", default=None,
                    help="name=MR,L,MC,KC,NC : bench one kernel across many N vs expert")
    args = ap.parse_args()
    cfgset = CONFIGS_R2 if args.round == "2" else CONFIGS

    if args.scaling:
        name, spec = args.scaling.split("=")
        MR, L, MC, KC, NC = [int(x) for x in spec.split(",")]
        Ns = (128, 256, 384, 512, 768, 1024)
        code = G.gen_kernel(MR=MR, L=L, MC=MC, KC=KC, NC=NC, pack=True)
        ok, log = build(name, code)
        assert ok, f"build failed: {log[-200:]}"
        cok, why = verify(name, vlens=(256, 128))
        assert cok, f"incorrect: {why}"
        kern = bench(name, Ns=Ns, reps=4)
        # expert (single-thread) across same N
        exp = expert_st(Ns, reps=3)
        out = {"kernel": name, "spec": dict(MR=MR, L=L, MC=MC, KC=KC, NC=NC),
               "ours": kern, "expert_st": exp,
               "pct": {N: 100 * kern[N] / exp[N] for N in Ns if exp.get(N)}}
        json.dump(out, open("_gemm_scaling.json", "w"), indent=1)
        print(f"\n{'N':>6}{'ours':>9}{'expert_st':>11}{'%exp':>7}")
        for N in Ns:
            print(f"{N:>6}{kern[N]:>9.3f}{exp.get(N,0):>11.3f}{out['pct'].get(N,0):>7.0f}")
        return

    expert = args.expertval if args.expertval else G.EXPERT_FALLBACK
    results = {}
    if args.expert:
        print("[expert] measuring OpenBLAS dgemm on board ...", flush=True)
        if build_expert():
            st = bench_existing("expert_dgemm", (512,), 3, env="OPENBLAS_NUM_THREADS=1")
            mt = bench_existing("expert_dgemm", (512,), 3, env="OPENBLAS_NUM_THREADS=8")
            expert = st[512]   # fair single-thread reference (paper uses 2.545)
            print(f"[expert] OpenBLAS dgemm N=512: single-thread={expert:.3f} "
                  f"multi-thread(8)={mt[512]:.3f} GFLOP/s", flush=True)
            results["__expert__"] = {"single_thread": st, "multi_thread": mt}

    cfgs = [(n, k) for n, k in cfgset if (args.only is None or args.only in n)]
    total = len(cfgs)
    best = 0.0
    bestname = None
    for i, (name, kw) in enumerate(cfgs, 1):
        code = G.gen_kernel(**kw)
        ok, log = build(name, code)
        if not ok:
            print(f"[{i}/{total}] {name}: BUILD_FAIL  {log.strip()[-160:]}", flush=True)
            results[name] = dict(status="buildfail", kw=kw)
            continue
        cok, why = verify(name)
        if not cok:
            print(f"[{i}/{total}] {name}: INCORRECT ({why})", flush=True)
            results[name] = dict(status="incorrect", why=why, kw=kw)
            continue
        b = bench(name, Ns=(512,))
        g = b[512]
        if g > best:
            best, bestname = g, name
        pct = 100 * g / expert
        results[name] = dict(status="ok", gflops=b, pct_expert=pct, kw=kw)
        print(f"[{i}/{total}] {name}: {g:.3f} GFLOP/s = {pct:.0f}% of expert "
              f"(best {best:.3f} via {bestname})", flush=True)

    results["__summary__"] = dict(expert=expert, best=best, bestname=bestname,
                                  pct=100 * best / expert if expert else 0)
    json.dump(results, open("_gemm_opt.json", "w"), indent=1)
    print(f"\nBEST = {best:.3f} GFLOP/s = {100*best/expert:.0f}% of expert ({expert:.3f}) "
          f"via {bestname}", flush=True)


if __name__ == "__main__":
    main()
