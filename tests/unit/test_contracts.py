import json
import pytest
from agentvec.air import AIR, Access, Alias, CPhy, Dep, Expr, Pattern, SAlgo, Status
from agentvec.air_from_formula import air_from_intent, parse_formula
from agentvec.guard import guard
from agentvec.lowering import emit_rvv_kernel, emit_reduce_kernel
from agentvec.pipeline import checked_air, lower_from_intent


@pytest.mark.parametrize("formula", ["out=2*a+b", "-a + 1e-3*b", "abs(a)", "out=.5*a", "a*b", "3"])
def test_supported_formulas(formula):
    air = air_from_intent(dict(pattern="map", dtype="f32", formula=formula))
    assert guard(air)[0]
    assert air.v_meta.status == Status.CHECKED
    assert "__riscv_vsetvl_e32m1" in emit_rvv_kernel(air)


@pytest.mark.parametrize("formula", ["a/b", "a;b", "a[1]", "a b", "out=a; out=b", "x=a", "sin(a)",
                                       "a**2", "__import__('os')", "(a", "", "a // b", "a^b", "True"])
def test_unsupported_syntax_is_not_silently_truncated(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


@pytest.mark.parametrize("mutation", [
    lambda a: setattr(a.c_phy, "alias", Alias.MAY_ALIAS),
    lambda a: setattr(a.c_phy, "dep", Dep.REDUCTION),
    lambda a: setattr(a.c_phy, "dtype", "f64"),
    lambda a: setattr(a.m_map, "access", Access.STRIDED),
    lambda a: setattr(a.m_map, "stride", 2),
    lambda a: setattr(a.m_map, "bound", "n+1"),
    lambda a: setattr(a.s_algo.elem_op, "src", "a;system(0)"),
])
def test_lowering_rechecks_even_a_forged_verified_record(mutation):
    air = AIR(SAlgo(Pattern.MAP, Expr("load", src="a")), CPhy("f32"))
    air.v_meta.status = Status.VERIFIED
    mutation(air)
    with pytest.raises(ValueError):
        emit_rvv_kernel(air)
    assert air.v_meta.status == Status.DRAFT


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e100, True])
def test_nonrepresentable_constants_are_rejected(value):
    air = AIR(SAlgo(Pattern.MAP, Expr("const", val=value)), CPhy("f32"))
    assert not guard(air)[0]


def test_reduction_preserves_expression_and_requires_reassociation_policy():
    intent = dict(pattern="reduce", dtype="f32", reduce_op="sum", elem="abs(a)")
    air = air_from_intent(intent, constraints=CPhy("f32", dep=Dep.REDUCTION, equiv="bitexact"))
    assert not guard(air)[0]
    air.c_phy.equiv = "tolerance"
    code = emit_rvv_kernel(air)
    assert "vfabs" in code
    assert "_tu(acc, acc," in code
    assert air.v_meta.status == Status.CHECKED


def test_json_round_trip_does_not_trust_serialized_status():
    air = air_from_intent(dict(pattern="reduce", dtype="f32", reduce_op="sum", elem="a*a", postproc="sqrt"))
    air.v_meta.status = Status.VERIFIED
    restored = AIR.from_json(air.to_json())
    assert restored.v_meta.status == Status.DRAFT
    assert restored.s_algo.postproc == "sqrt"
    assert restored.s_algo.elem_op.to_dict() == air.s_algo.elem_op.to_dict()


@pytest.mark.parametrize("op", ["dot", "asum", "saxpy"])
def test_missing_model_answer_never_falls_back_to_correct_intent(op):
    with pytest.raises(ValueError):
        lower_from_intent(op, None)


def test_proposal_cannot_override_source_rules():
    with pytest.raises(ValueError):
        checked_air("asum", dict(pattern="map", dtype="f32", formula="abs(a)"))
    with pytest.raises(ValueError):
        checked_air("copy", dict(pattern="map", dtype="i32", formula="a"))
    with pytest.raises(ValueError):
        air_from_intent(dict(pattern="map", dtype="f32", formula="a", alias="disjoint"))


@pytest.mark.parametrize("op,element,post", [("mean", "a", None), ("sum", "unknown", None), ("sum", "a", "square")])
def test_reduce_compatibility_api_cannot_bypass_validation(op, element, post):
    with pytest.raises(ValueError):
        emit_reduce_kernel(op, element, post)
