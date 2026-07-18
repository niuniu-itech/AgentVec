"""Closed-loop performance refinement for AgentVec's GEMM lowering (the future-work
made concrete). The deterministic lowering is PARAMETERIZED (LMUL, loop strategy,
block sizes); a feedback loop compiles each config, verifies correctness, measures
GFLOP/s on the K1, and keeps the best-so-far. We plot best-so-far vs iteration to
see when it converges and how close it gets to the hand-written expert (2.5 GFLOP/s).
Determinism is preserved: a fixed config still yields a fixed kernel; refinement only
SELECTS the config using on-hardware feedback. [i/N] progress."""
import os, sys, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get, HOSTS
DST = "/path/to/operator_extract"; GCC = f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"
QEMU = f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"; work = "/path/to/agentvec_lab/gemm_l3"
EXPERT = 2.545  # measured OpenBLAS expert GFLOP/s @ N=512


def rank1(L):
    return f"""#include <riscv_vector.h>
void gemm(int N,const double*A,const double*B,double*C){{
  for(int i=0;i<N;i++){{ for(int j=0;j<N;j++)C[i*N+j]=0.0;
    for(int k=0;k<N;k++){{ double a=A[(size_t)i*N+k]; const double*Br=&B[(size_t)k*N]; double*Cr=&C[(size_t)i*N];
      for(int j=0;j<N;){{ size_t vl=__riscv_vsetvl_e64m{L}((size_t)(N-j));
        vfloat64m{L}_t vc=__riscv_vle64_v_f64m{L}(Cr+j,vl),vb=__riscv_vle64_v_f64m{L}(Br+j,vl);
        vc=__riscv_vfmacc_vf_f64m{L}(vc,a,vb,vl); __riscv_vse64_v_f64m{L}(Cr+j,vc,vl); j+=(int)vl; }}}}}}}}"""


def blocked(L, BK, BJ):
    return f"""#include <riscv_vector.h>
#define BK {BK}
#define BJ {BJ}
void gemm(int N,const double*A,const double*B,double*C){{
  for(size_t t=0;t<(size_t)N*N;t++)C[t]=0.0;
  for(int jj=0;jj<N;jj+=BJ){{ int jm=jj+BJ<N?jj+BJ:N;
    for(int kk=0;kk<N;kk+=BK){{ int km=kk+BK<N?kk+BK:N;
      for(int i=0;i<N;i++){{ const double*Ar=&A[(size_t)i*N]; double*Cr=&C[(size_t)i*N];
        for(int k=kk;k<km;k++){{ double a=Ar[k]; const double*Br=&B[(size_t)k*N];
          for(int j=jj;j<jm;){{ size_t vl=__riscv_vsetvl_e64m{L}((size_t)(jm-j));
            vfloat64m{L}_t vc=__riscv_vle64_v_f64m{L}(Cr+j,vl),vb=__riscv_vle64_v_f64m{L}(Br+j,vl);
            vc=__riscv_vfmacc_vf_f64m{L}(vc,a,vb,vl); __riscv_vse64_v_f64m{L}(Cr+j,vc,vl); j+=(int)vl; }}}}}}}}}}}}"""


# search order: start from the deterministic default (rank-1 m1), then explore.
CONFIGS = [
    ("rank1 m1 (default)", rank1(1)),
    ("rank1 m2", rank1(2)),
    ("rank1 m4", rank1(4)),
    ("rank1 m8", rank1(8)),
    ("block m2 64x128", blocked(2, 64, 128)),
    ("block m4 64x128", blocked(4, 64, 128)),
    ("block m4 128x256", blocked(4, 128, 256)),
    ("block m8 64x256", blocked(8, 64, 256)),
    ("block m8 128x256", blocked(8, 128, 256)),
    ("block m4 256x256", blocked(4, 256, 256)),
    ("block m8 256x512", blocked(8, 256, 512)),
    ("block m8 512x512", blocked(8, 512, 512)),
]


def main():
    rows = []; best = 0.0; total = len(CONFIGS)
    bdir = "/home/user/agentvec_lab/autotune"; run("board", f"mkdir -p {bdir}")
    for i, (name, code) in enumerate(CONFIGS, 1):
        local = f"_gen/at_{i}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(code)
        put("server", local, f"{work}/at_{i}.c")
        # build correctness + perf
        cc = run("server", f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c at_{i}.c -o atc_{i}.out 2>&1 && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static perf.c at_{i}.c -o atp_{i}.out 2>&1 && echo OK", timeout=180)[1]
        if "OK" not in cc:
            print(f"[{i}/{total}] {name}: BUILD_FAIL", flush=True); rows.append((name, "buildfail", 0.0, best)); continue
        corr = run("server", f"cd {work} && {QEMU} -cpu rv64,v=true,vlen=256,vext_spec=v1.0 atc_{i}.out 2>&1 | grep SUMMARY", timeout=180)[1]
        ok = bool(re.search(r"fails=0/15", corr))
        if not ok:
            print(f"[{i}/{total}] {name}: INCORRECT", flush=True); rows.append((name, "incorrect", 0.0, best)); continue
        get("server", f"{work}/atp_{i}.out", f"_board/atp_{i}.out"); put("board", f"_board/atp_{i}.out", f"{bdir}/run.out"); run("board", f"chmod +x {bdir}/run.out")
        gf = []
        for _ in range(2):
            o = run("board", f"cd {bdir} && ./run.out 512 2>&1", timeout=300)[1]
            m = re.search(r"GFLOPS=([0-9.]+)", o); gf.append(float(m.group(1)) if m else 0.0)
        g = max(gf); best = max(best, g)
        rows.append((name, "ok", g, best))
        print(f"[{i}/{total}] {name}: {g:.3f} GFLOP/s  (best-so-far {best:.3f} = {100*best/EXPERT:.0f}% of expert)", flush=True)
    json.dump([{"name": n, "status": s, "gflops": g, "best": b} for n, s, g, b in rows], open("_autotune.json", "w"))
    print(f"\nCONVERGED best={best:.3f} GFLOP/s = {100*best/EXPERT:.0f}% of expert ({EXPERT})")


if __name__ == "__main__":
    main()
