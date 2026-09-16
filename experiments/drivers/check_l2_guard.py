"""Illustrative L2 contracts. These are not full L2 AIRs or a lowering test.

The core backend rejects f64 here; dedicated BLAS emitters require their own
operator-specific checks. A simplified map/reduce skeleton does not establish
matrix-address or dependence correctness for an entire BLAS operation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from agentvec.air import AIR, SAlgo, CPhy, MMap, Expr, Pattern, Dep, Alias, Access
from agentvec.guard import guard

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
        print(f"  {'CHECKED ' if ok else 'VETO    '}  {name:24s} :: {msg}")
