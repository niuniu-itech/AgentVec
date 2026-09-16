"""Validate registered Ascend intents before admitting code generation."""
from dataclasses import asdict

from .air import AIR, Pattern, Status, VMeta


def guard(air: AIR):
    from .registry import ALL_OPS, recover
    air.v_meta = VMeta()
    reason = ""
    if air.name not in ALL_OPS:
        reason = "operator is not in the registered Ascend source family"
    elif air.s_algo.pattern in (Pattern.TRSV, Pattern.TRSM):
        reason = "triangular solve has a loop-carried dependence; no parallel template is admitted"
    elif air.c_phy.dtype != "f32":
        reason = "published Ascend templates implement f32 only"
    else:
        reference = recover(air.name)
        if asdict(air.c_phy) != asdict(reference.c_phy):
            reason = "type, dependence, alias or equivalence differs from the source contract"
        elif asdict(air.m_map) != asdict(reference.m_map):
            reason = "layout, stride or bounds differs from the registered source contract"
        elif asdict(air.s_algo) != asdict(reference.s_algo):
            reason = "intent does not match the supported registered template semantics"
    if reason:
        air.v_meta.proof = reason
        return False, reason
    air.v_meta = VMeta(Status.CHECKED, "registered intent, f32 rules, layout and dependency checks passed")
    return True, air.v_meta.proof
