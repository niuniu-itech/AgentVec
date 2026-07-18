"""Symbolic Guard verdicts for the L2 operators (deterministic, no LLM).

Confirms the Guard ADMITS the four vectorizable L2 ops (map / associative reduction,
no loop-carried dependence) -- consistent with their measured expert parity on hardware
-- and conservatively REJECTS dtrsv, whose triangular solve carries a loop-carried
dependence (x[i] reads x[<i]); AgentVec then falls back to the scalar build rather than
emitting a silently-wrong parallelized kernel."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from air import AIR, SAlgo, CPhy, MMap, Expr, Pattern, Dep, Alias, Access
from guard import guard

CASES = {
    "dgemv (y = A x)":        AIR(SAlgo(Pattern.REDUCE, reduce_op="sum", reduce_src="a"),
                                  CPhy("f64", dep=Dep.REDUCTION), MMap(Access.UNIT_STRIDE)),
    "dger  (A += a x y^T)":   AIR(SAlgo(Pattern.MAP, elem_op=Expr("fma", args=[
                                      Expr("load", src="x"), Expr("load", src="y"), Expr("load", src="a")])),
                                  CPhy("f64", dep=Dep.NONE, alias=Alias.DISJOINT), MMap(Access.UNIT_STRIDE)),
    "dsymv (sym y = A x)":    AIR(SAlgo(Pattern.REDUCE, reduce_op="sum", reduce_src="a"),
                                  CPhy("f64", dep=Dep.REDUCTION), MMap(Access.UNIT_STRIDE)),
    "dspmv (packed sym y=Ax)":AIR(SAlgo(Pattern.REDUCE, reduce_op="sum", reduce_src="a"),
                                  CPhy("f64", dep=Dep.REDUCTION), MMap(Access.UNIT_STRIDE)),
    "dtrsv (solve L x = b)":  AIR(SAlgo(Pattern.MAP, elem_op=Expr("load", src="b")),
                                  CPhy("f64", dep=Dep.LOOP_CARRIED, dep_distance=1), MMap(Access.UNIT_STRIDE)),
}

if __name__ == "__main__":
    print("Symbolic Guard verdicts (deterministic dependence/type analysis):\n")
    for name, air in CASES.items():
        ok, msg = guard(air)
        print(f"  {'VERIFIED' if ok else 'VETO    '}  {name:24s} :: {msg}")
