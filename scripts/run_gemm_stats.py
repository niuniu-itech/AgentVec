"""GEMM GFLOP/s vs N, REPS repeats per point (board K1), median/min/max for error bars."""
import sys,re,statistics,json; sys.path.insert(0,"../lab")
from ssh_helper import run
REPS=5; NS=[64,128,256,384,512]; KERNS=["agentvec","gpt","opus"]
out={}
for kern in KERNS:
    bd=f"/home/user/agentvec_lab/perf_{kern}"; out[kern]={}
    for N in NS:
        vals=[]
        for _ in range(REPS):
            o=run("board",f"cd {bd} && ./run.out {N} 2>&1",timeout=300)[1]
            m=re.search(r"GFLOPS=([0-9.]+)",o)
            if m: vals.append(float(m.group(1)))
        if vals:
            out[kern][N]=dict(median=statistics.median(vals),mn=min(vals),mx=max(vals),std=(statistics.pstdev(vals) if len(vals)>1 else 0))
            print(f"{kern:9} N={N:4} median={statistics.median(vals):.3f} [{min(vals):.3f},{max(vals):.3f}] std={statistics.pstdev(vals):.3f}",flush=True)
json.dump(out,open("_gemm_stats.json","w"))
