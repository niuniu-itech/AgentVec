"""Run the already-built benchmark binaries (AgentVec + expert) natively on the
K1 board for real (non-qemu) performance. Binaries are static; pull from server,
push to board, run with averaging."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, get, put

OPS = ["daxpy", "dscal", "dcopy", "ddot", "dasum"]
RNG = "2048 16384 2048"      # larger sizes -> stable timing on real hardware
LOOPS = "100"
LOC = os.path.join(os.path.dirname(__file__), "_board")
os.makedirs(LOC, exist_ok=True)


def fetch_run(op, kind, server_out):
    lp = os.path.join(LOC, f"{op}_{kind}.out")
    get("server", server_out, lp)
    bdir = f"/home/user/agentvec_lab/{op}_{kind}"
    run("board", f"mkdir -p {bdir}")
    put("board", lp, f"{bdir}/run.out")
    rc, out, err = run("board", f"cd {bdir} && chmod +x run.out && "
                                 f"OPENBLAS_LOOPS={LOOPS} ./run.out {RNG} 2>&1", timeout=300)
    res = re.findall(r"=>\s*(pass|fail)", out)
    sp = re.search(r"Average Speedup:\s*([0-9.]+)", out)
    return (sp.group(1) if sp else "?"), res.count("pass"), len(res), out


print(f"{'op':8}{'AV_sp':>8}{'exp_sp':>8}{'MS':>8}{'AV pass':>10}")
for op in OPS:
    av_sp, ap, at, _ = fetch_run(op, "av", f"/path/to/agentvec_lab/runs/{op}/{op}.out")
    ex_sp, ep, et, _ = fetch_run(op, "exp", f"/path/to/agentvec_lab/expert/{op}/{op}.out")
    try:
        ms = round(float(av_sp) / float(ex_sp), 3)
    except Exception:
        ms = "?"
    print(f"{op:8}{av_sp:>8}{ex_sp:>8}{str(ms):>8}{str(ap)+'/'+str(at):>10}")
