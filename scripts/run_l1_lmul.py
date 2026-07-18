"""L1 LMUL refinement: measure MS vs expert for axpy/asum/dot at LMUL m1/m2/m4/m8
through the real OpenBLAS benchmark (robust, averaged over sizes). Shows the default
m1 below expert climbing to parity as LMUL is tuned -- the refinement-reaches-parity story."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get
from blas_lower import emit_l1, L1Spec
from run_l1_board import build_run, OPB

OPS = [("daxpy", "13", "axpy"), ("dasum", "11", "asum"), ("ddot", "15", "dot")]
LMULS = [1, 2, 4, 8]


def main():
    print(f"{'op':8}" + "".join(f"{'m'+str(L):>9}" for L in LMULS), flush=True)
    out = {}
    for op, num, kind in OPS:
        base = emit_l1(L1Spec(name=f"{op}_k_rvv", dtype="f64", kind=kind))  # m8 version
        exp, _ = build_run(op, num, f"{OPB}/{num}_{op}_op/{op}_k_rvv.c", "exp")
        row = {}
        for L in LMULS:
            src = base.replace("m8", f"m{L}")  # retarget LMUL; m1 tokens (reduce dest) unaffected
            local = f"_gen/{op}_m{L}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(src)
            work = f"/path/to/agentvec_lab/l1lmul/{op}_m{L}"; run("server", f"mkdir -p {work}"); put("server", local, f"{work}/{op}_k_rvv.c")
            av, _ = build_run(op, num, f"{work}/{op}_k_rvv.c", f"m{L}")
            ms = (av[0] / exp[0]) if (av and exp and av[0] and exp[0]) else None
            row[L] = ms
        out[op] = (exp[0] if exp else None, row)
        print(f"{op:8}" + "".join(f"{(row[L] if row[L] else 0):>9.3f}" for L in LMULS), flush=True)
    import json; json.dump({k: {"expert_sp": v[0], "ms": v[1]} for k, v in out.items()}, open("_l1_lmul.json", "w"))


if __name__ == "__main__":
    main()
