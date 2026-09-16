"""Contracts and semantic boundaries of the general map/reduction graph path."""
from dataclasses import asdict, replace
import json
import pytest

from agentvec.air import Status
from agentvec.dag import DagAIR, DagContract, DagNode, check_dag, expression
from agentvec.dag_examples import EXAMPLES, example
from agentvec.dag_lowering import DagSchedule, emit_dag


def graph(nodes, inputs=None, outputs=None):
    return DagAIR.from_proposal({"pattern": "dag", "nodes": nodes},
                               DagContract(inputs or {"x": "n"}, outputs or {"y": "n"}))


def node(name, formula, reduce=None):
    result = dict(name=name, pattern="reduce" if reduce else "map", formula=formula)
    if reduce:
        result["reduce_op"] = reduce
    return result


@pytest.mark.parametrize("name", EXAMPLES)
def test_registered_graph_shapes_and_independent_reference(name):
    g = example(name)
    checked = check_dag(g)
    assert checked.shapes["y"] == "n"
    assert g.v_meta.status == Status.CHECKED
    clone = DagAIR.from_proposal({"pattern":"dag", "nodes":[asdict(n) for n in g.nodes]}, g.contract)
    assert clone.fingerprint() == g.fingerprint()


def test_arbitrary_branch_join_and_multiple_outputs():
    g = graph([node("right", "x*gain"), node("y", "left+right"),
               node("left", "x+1"), node("total", "y", "sum")],
              {"x":"n", "gain":"scalar"}, {"y":"n", "total":"scalar"})
    checked = check_dag(g)
    assert [n.name for n in checked.order].index("left") < [n.name for n in checked.order].index("y")
    source, record = emit_dag(g)
    assert record["status"] == "CHECKED"
    assert "__riscv_vfmv_v_f_f32m1(inputs[1][0], vl)" in source
    assert "outputs[1]" in source


@pytest.mark.parametrize("nodes,reason", [
    ([node("x","x+1"),node("y","x")], "SSA"),
    ([node("a","x+1"),node("a","x+2"),node("y","a")], "SSA"),
    ([node("a","y+1"),node("y","a+1")], "cyclic"),
    ([node("y","missing+1")], "undefined"),
    ([node("a","x+1"),node("y","x+2")], "dead"),
    ([node("y","x","sum")], "extent"),
    ([node("a","1","sum"),node("y","x+a")], "vector"),
])
def test_cross_node_hazards_rejected(nodes, reason):
    with pytest.raises(ValueError, match=reason):
        graph(nodes)


@pytest.mark.parametrize("formula", ["x[0]", "x; y", "__import__('os')", "x.__class__",
                                    "exp(x,1)", "pow(x,2)", "True", "1e400", "(lambda: 0)()"])
def test_expression_language_is_closed(formula):
    with pytest.raises(ValueError):
        expression(formula)


def test_serialized_verification_and_contract_cannot_be_supplied_by_proposer():
    with pytest.raises(ValueError):
        DagAIR.from_proposal(dict(pattern="dag", nodes=[node("y","x")], status="VERIFIED"),
                             DagContract({"x":"n"},{"y":"n"}))
    g = example("softmax")
    g.v_meta.status = Status.VERIFIED
    g.contract = replace(g.contract, dtype="f64")
    with pytest.raises(ValueError, match="f32"):
        emit_dag(g)
    assert g.v_meta.status == Status.DRAFT


def test_reassociation_and_output_alias_rejected():
    g = example("softmax")
    g.nodes = tuple(replace(n,equiv="bitexact") if n.reduce_op=="sum" else n for n in g.nodes)
    with pytest.raises(ValueError, match="reassociation"):
        check_dag(g)
    with pytest.raises(ValueError, match="alias"):
        graph([node("y","x")], outputs={"x":"n"})


def test_fusion_keeps_branched_producers_but_removes_single_use_map():
    g = example("softmax")
    _, record = emit_dag(g, schedule=DagSchedule(True))
    assert "shifted" in record["topological_order"]
    g = example("rmsnorm")
    before, _ = emit_dag(g)
    after, record = emit_dag(g, schedule=DagSchedule(True))
    assert ["squares","energy"] in record["fused_edges"]
    assert before != after


def test_unsupported_schedule_and_target_are_rejected():
    with pytest.raises(ValueError):
        emit_dag(example("softmax"), target="cuda")
    with pytest.raises(ValueError):
        emit_dag(example("softmax"), schedule=DagSchedule("true"))


def test_nested_exp_has_distinct_lane_buffers():
    source, _ = emit_dag(graph([node("y","exp(exp(x))")]))
    import re
    names = re.findall(r"float (lane\d+)\[vl\]", source)
    assert len(names) == len(set(names)) == 2
