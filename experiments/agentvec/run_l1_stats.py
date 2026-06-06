"""Statistical re-measurement for the L1 parity claim (figure6a) and GEMM (figure9_l3):
each measurement is repeated REPS times (whole board run, not just the benchmark's internal
averaging), and we report median / min / max so the paper can show error bars and a methodology.
Hardware/flags are recorded for the reproducibility paragraph."""
import os, sys, re, statistics, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get
from blas_lower import emit_l1, L1Spec
from run_l1_board import build_run, OPB
REPS=5
OPS=[("daxpy","13","axpy"),("dscal","24","scal"),("dcopy","14","copy"),("ddot","15","dot"),
     ("dasum","11","asum"),("daxpby","12","axpby"),("dswap","28","swap"),("dmax","20","max"),("dmin","21","min")]

def main():
    print(f"=== L1 MS (AgentVec/expert), {REPS} repeats each, board K1 ===",flush=True)
    rows={}
    for op,num,kind in OPS:
        ksrc=emit_l1(L1Spec(name=f"{op}_k_rvv",dtype="f64",kind=kind))
        local=f"_gen/{op}_k_rvv.c"; os.makedirs("_gen",exist_ok=True); open(local,"w",newline="\n").write(ksrc)
        work=f"/path/to/agentvec_lab/l1stats/{op}"; run("server",f"mkdir -p {work}"); put("server",local,f"{work}/{op}_k_rvv.c")
        ms=[]
        for r in range(REPS):
            av,_=build_run(op,num,f"{work}/{op}_k_rvv.c","av")
            ex,_=build_run(op,num,f"{OPB}/{num}_{op}_op/{op}_k_rvv.c","exp")
            if av and ex and av[0] and ex[0]: ms.append(av[0]/ex[0])
        if ms:
            rows[op]=dict(median=statistics.median(ms),mn=min(ms),mx=max(ms),
                          std=(statistics.pstdev(ms) if len(ms)>1 else 0.0),n=len(ms),vals=[round(x,3) for x in ms])
            print(f"  {op:7} MS median={rows[op]['median']:.3f}  min={rows[op]['mn']:.3f} max={rows[op]['mx']:.3f} std={rows[op]['std']:.3f}  {rows[op]['vals']}",flush=True)
    print(f"\nmean of medians = {statistics.mean(r['median'] for r in rows.values()):.3f}",flush=True)
    json.dump(rows,open("_l1_stats.json","w"))

if __name__=="__main__": main()
