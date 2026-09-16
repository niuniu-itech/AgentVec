"""Check the contracts supported by the unit-stride core backend.

Rules are supplied independently of the proposer. CHECKED permits lowering;
only a subsequent build and differential test can establish VERIFIED.
This module does not extract dependence facts from arbitrary C or assembly.
"""
from .air import AIR, Pattern, Dep, Alias, Access, Status


def guard(air: AIR) -> tuple[bool, str]:
    air.v_meta.status = Status.DRAFT
    ok, message = air.validate_schema()
    if ok:
        if air.c_phy.dtype not in ("f32", "i32"):
            ok, message = False, "core supports f32/i32; f64 BLAS uses separate templates"
        elif air.m_map.access != Access.UNIT_STRIDE or air.m_map.stride != 1 or air.m_map.bound != "n":
            ok, message = False, "core requires unit-stride accesses over 0 <= i < n"
        elif air.c_phy.alias != Alias.DISJOINT:
            ok, message = False, "unresolved aliasing: disjoint input/output storage is required"
        elif air.s_algo.pattern == Pattern.MAP and air.c_phy.dep != Dep.NONE:
            ok, message = False, "MAP requires no carried dependence"
        elif air.s_algo.pattern == Pattern.REDUCE and air.c_phy.dep not in (Dep.NONE, Dep.REDUCTION):
            ok, message = False, "reduction has an unsupported carried dependence"
        elif (air.s_algo.pattern == Pattern.REDUCE and air.c_phy.dtype == "f32"
              and air.s_algo.reduce_op == "sum" and air.c_phy.equiv == "bitexact"):
            ok, message = False, "FP sum reassociation requires a tolerance policy"
        else:
            message = "schema, type, dependence, alias and unit-stride bounds checked"
    air.v_meta.status = Status.CHECKED if ok else Status.DRAFT
    air.v_meta.proof = message
    return ok, message


def feature_select(candidates: list[AIR]) -> AIR | None:
    """Return the first candidate that passes a fresh contract check."""
    return next((candidate for candidate in candidates if guard(candidate)[0]), None)
