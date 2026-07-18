"""Phase 2: the Symbolic Guard — a DETERMINISTIC checker (NOT an LLM).

Given an AIR whose S_algo/M_map were proposed by the Intent Proposer and whose
C_phy was produced independently by static dependence/type analysis (Track beta),
the Guard decides VERIFIED or applies a Safety Veto. Correctness of the verdict
does not depend on the language model (this answers the reviewer's objection that
"if the Guard is an LLM it cannot guarantee correctness").

v1 implements the legality rules a polyhedral/loop-dependence analysis yields for
the supported patterns; it is the checking half. Track-beta extraction (computing
C_phy from source) is a separate component; here C_phy is the trusted input the
Guard checks the hypothesis against. The Guard is conservative: anything it cannot
prove safe is vetoed (fallback to a safe form), never optimistically accepted.
"""
from air import AIR, Pattern, Dep, Alias, Access, Status


def _check_types(air: AIR) -> tuple[bool, str]:
    if air.c_phy.dtype not in ("f32", "i32", "f64"):
        return False, f"unsupported dtype {air.c_phy.dtype}"
    if air.s_algo.pattern == Pattern.MAP and air.s_algo.elem_op is None:
        return False, "MAP without elem_op"
    if air.s_algo.pattern == Pattern.REDUCE and air.s_algo.reduce_op not in ("sum", "max", "min"):
        return False, f"REDUCE op {air.s_algo.reduce_op} not associative-recognized"
    return True, "types consistent"


def _check_dependence(air: AIR) -> tuple[bool, str]:
    """Decide parallel/vectorization legality from C_phy + pattern + access."""
    dep = air.c_phy.dep
    if air.s_algo.pattern == Pattern.MAP:
        # Elementwise map (incl. in-place y[i]=f(y[i],x[i])) is parallel iff no
        # loop-carried dependence and the write set does not alias a *shifted*
        # read of the same array. A unit/strided affine self-map at distance 0
        # carries no dependence.
        if dep == Dep.LOOP_CARRIED and air.c_phy.dep_distance != 0:
            return False, f"VETO: loop-carried dependence (distance={air.c_phy.dep_distance}) forbids MAP vectorization"
        if air.c_phy.alias == Alias.MAY_ALIAS:
            return False, "VETO: may-alias between in/out; cannot prove disjoint access"
        if air.m_map.access == Access.INDEXED:
            return False, "VETO: indexed access under MAP needs gather (use GATHER_APPLY)"
        return True, "no loop-carried dependence at distance>0; affine disjoint access => parallel-safe"
    if air.s_algo.pattern == Pattern.REDUCE:
        # The only carried dependence is the accumulator; a vector reduction over
        # an associative op is legal (with FP reassociation accepted under the ULP tier).
        if dep not in (Dep.REDUCTION, Dep.NONE):
            return False, f"VETO: REDUCE expected reduction/none dependence, got {dep.value}"
        return True, f"associative {air.s_algo.reduce_op} reduction; vector reduction legal"
    return False, f"VETO: pattern {air.s_algo.pattern.value} not supported by lowering"


def guard(air: AIR) -> tuple[bool, str]:
    """Run the deductive checks; mutate v_meta accordingly. Return (verified, proof)."""
    ok, msg = air.validate_schema()
    if not ok:
        air.v_meta.status = Status.DRAFT; air.v_meta.proof = f"schema: {msg}"
        return False, air.v_meta.proof
    ok, tmsg = _check_types(air)
    if not ok:
        air.v_meta.status = Status.DRAFT; air.v_meta.proof = tmsg
        return False, tmsg
    air.v_meta.status = Status.CHECKED
    ok, dmsg = _check_dependence(air)
    if not ok:
        air.v_meta.proof = dmsg
        return False, dmsg
    air.v_meta.status = Status.VERIFIED
    air.v_meta.proof = f"{tmsg}; {dmsg}"
    return True, air.v_meta.proof


def feature_select(candidates: list[AIR]) -> AIR | None:
    """Among VERIFIED candidates, prefer the most target-friendly access pattern."""
    verified = [c for c in candidates if c.v_meta.status == Status.VERIFIED]
    if not verified:
        return None
    rank = {Access.UNIT_STRIDE: 0, Access.STRIDED: 1, Access.INDEXED: 2}
    return min(verified, key=lambda c: rank.get(c.m_map.access, 9))


if __name__ == "__main__":
    from air import SAlgo, CPhy, MMap, Expr, Pattern as P
    # 1) valid saxpy -> VERIFIED
    ok, msg = guard(AIR(s_algo=SAlgo(pattern=P.MAP,
                   elem_op=Expr("add", args=[Expr("mulc", val=2.0, args=[Expr("load", src="a")]),
                                             Expr("load", src="b")])),
                   c_phy=CPhy(dtype="f32", dep=Dep.NONE, alias=Alias.DISJOINT)))
    print("saxpy:", ok, "|", msg)
    # 2) MAP but loop-carried dep -> VETO
    ok, msg = guard(AIR(s_algo=SAlgo(pattern=P.MAP, elem_op=Expr("load", src="a")),
                   c_phy=CPhy(dtype="f32", dep=Dep.LOOP_CARRIED, dep_distance=1)))
    print("loop-carried:", ok, "|", msg)
    # 3) may-alias -> VETO
    ok, msg = guard(AIR(s_algo=SAlgo(pattern=P.MAP, elem_op=Expr("load", src="a")),
                   c_phy=CPhy(dtype="f32", dep=Dep.NONE, alias=Alias.MAY_ALIAS)))
    print("may-alias:", ok, "|", msg)
    # 4) reduction -> VERIFIED
    ok, msg = guard(AIR(s_algo=SAlgo(pattern=P.REDUCE, reduce_op="sum", reduce_src="a"),
                   c_phy=CPhy(dtype="f32", dep=Dep.REDUCTION)))
    print("fsum:", ok, "|", msg)
