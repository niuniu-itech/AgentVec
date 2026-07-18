"""HARD PROBLEM 3, part 1 -- baseline B (symbolic-only pruning).
Run the full inflated schedule space A once on the real K1; B is the subset the
symbolic Align admits (a FILTER on A's results, so A is measured once, B is free).
Report: |A| vs |B|, and whether B still contains A's optimum (= pruning didn't lose
the best). This single number decides Agentic-dominant vs Symbolic-dominant narrative.
Run:  python -u run_baseline_B.py   (uses board+server via ssh_helper)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
import gemm_opt as G
import run_gemm_opt as R
from symbolic_align import align, H_K1, I_GEMM

LMUL = [1, 2, 4, 8]; PACK = [True, False]; MR = [2, 4, 6, 8]; KC = [128, 256, 512]
MC, NC, N = 120, 240, 512


def cfgs():
    return [{"L": L, "pack": pk, "MR": mr, "KC": kc, "MC": MC, "NC": NC}
            for L in LMUL for pk in PACK for mr in MR for kc in KC]


def tag(c):
    return f"B_L{c['L']}_pk{int(c['pack'])}_mr{c['MR']}_kc{c['KC']}"


def gen(c):
    return G.gen_kernel(MR=c["MR"], L=c["L"], MC=c["MC"], KC=c["KC"], NC=c["NC"], pack=c["pack"])


def main():
    A = cfgs()
    results = {}
    # 1) correctness on a representative sample (template is correct by construction;
    #    confirm across lmul/MR/packing ranges)
    sample = [c for c in A if c["pack"] and c["L"] in (1, 4) and c["MR"] in (2, 6) and c["KC"] == 256]
    print(f"=== verifying {len(sample)} representative configs (QEMU multi-VLEN) ===", flush=True)
    for c in sample:
        t = tag(c) + "_v"
        ok, _ = R.build(t, gen(c))
        cok, why = R.verify(t) if ok else (False, "buildfail")
        print(f"  {tag(c)}: build={ok} correct={cok} {why}", flush=True)
    # 2) full sweep: build + bench @ N=512
    print(f"\n=== sweeping |A|={len(A)} configs (build + bench @N={N}) ===", flush=True)
    for i, c in enumerate(A, 1):
        t = tag(c)
        inB, reason = align(c, I_GEMM, H_K1)
        ok, log = R.build(t, gen(c))
        if not ok:
            results[t] = {"cfg": c, "status": "buildfail", "in_B": inB, "reason": reason}
            print(f"[{i}/{len(A)}] {t}: BUILD_FAIL {'[B]' if inB else '[pruned]'}", flush=True)
            continue
        g = R.bench(t, Ns=(N,), reps=2).get(N, 0.0)
        results[t] = {"cfg": c, "status": "ok", "gflops": g, "in_B": inB, "reason": reason}
        print(f"[{i}/{len(A)}] {t}: {g:6.3f} GFLOP/s  {'[B]' if inB else '[pruned: '+reason[:34]+']'}", flush=True)

    okc = {t: r for t, r in results.items() if r["status"] == "ok"}
    B_ok = {t: r for t, r in okc.items() if r["in_B"]}
    A_opt_t = max(okc, key=lambda t: okc[t]["gflops"]) if okc else None
    B_opt_t = max(B_ok, key=lambda t: B_ok[t]["gflops"]) if B_ok else None
    A_opt = okc[A_opt_t]["gflops"] if A_opt_t else 0.0
    B_opt = B_ok[B_opt_t]["gflops"] if B_opt_t else 0.0
    n_B_enum = sum(1 for c in A if align(c, I_GEMM, H_K1)[0])
    summary = {
        "A_enumerated": len(A), "A_built_ok": len(okc),
        "A_buildfail": sum(1 for r in results.values() if r["status"] == "buildfail"),
        "B_enumerated": n_B_enum, "B_built_ok": len(B_ok),
        "A_opt_gflops": round(A_opt, 3), "A_opt_cfg": A_opt_t,
        "B_opt_gflops": round(B_opt, 3), "B_opt_cfg": B_opt_t,
        "B_keeps_optimum": (A_opt > 0 and abs(A_opt - B_opt) <= 0.03 * A_opt),
    }
    results["__summary__"] = summary
    json.dump(results, open(os.path.join(os.path.dirname(__file__), "_baseline_B.json"), "w"), indent=1)
    print("\n==================== BASELINE B (symbolic-only) ====================")
    print(f"  A inflated : {len(A)} enumerated | {len(okc)} built-ok | {summary['A_buildfail']} buildfail")
    print(f"  B symbolic : {n_B_enum} enumerated | {len(B_ok)} built-ok   (prune {len(A)}->{n_B_enum})")
    print(f"  A optimum  : {A_opt:6.3f} GFLOP/s  [{A_opt_t}]")
    print(f"  B optimum  : {B_opt:6.3f} GFLOP/s  [{B_opt_t}]")
    print(f"  => B {'KEEPS' if summary['B_keeps_optimum'] else 'LOSES'} the optimum "
          f"while searching {n_B_enum}/{len(A)} of the space")
    print("saved _baseline_B.json")


if __name__ == "__main__":
    main()
