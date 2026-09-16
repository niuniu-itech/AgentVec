"""Typed, single-assignment AIR graphs with caller-owned contracts.

Graphs contain maps, scalar maps and vector reductions. Inputs and outputs share
a symbolic vector extent n or a scalar extent. No source dependence facts are
inferred here: the caller supplies the storage and numerical contract.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
import math
import re

from .air import Status, VMeta


@dataclass(frozen=True)
class DagExpr:
    op: str
    args: tuple[DagExpr, ...] = ()
    name: str = ""
    value: float = 0.0

    def reads(self):
        return ({self.name} if self.op == "read" else set()).union(
            *(child.reads() for child in self.args))


def expression(text: str) -> DagExpr:
    """Parse the whole expression; arbitrary calls and indexing are excluded."""
    if not isinstance(text, str) or not text or len(text) > 4096:
        raise ValueError("expected an expression of 1..4096 characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid graph expression") from exc

    def visit(node, depth=0):
        if depth > 32:
            raise ValueError("graph expression exceeds depth 32")
        if isinstance(node, ast.Name):
            return DagExpr("length") if node.id == "n" else DagExpr("read", name=node.id)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 3.402823466e38:
                raise ValueError("constant must be finite and representable in f32")
            return DagExpr("constant", value=float(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            child = visit(node.operand, depth + 1)
            return child if isinstance(node.op, ast.UAdd) else DagExpr("neg", (child,))
        operations = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return DagExpr(operations[type(node.op)], (visit(node.left, depth + 1), visit(node.right, depth + 1)))
        functions = {"exp": 1, "sqrt": 1, "abs": 1, "min": 2, "max": 2}
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == functions[node.func.id]
                and not node.keywords):
            return DagExpr(node.func.id, tuple(visit(a, depth + 1) for a in node.args))
        raise ValueError("unsupported DAG expression operation")

    return visit(tree.body)


@dataclass(frozen=True)
class DagNode:
    name: str
    pattern: str
    formula: str
    reduce_op: str | None = None
    equiv: str = "tolerance"


@dataclass(frozen=True)
class DagContract:
    inputs: dict[str, str]
    outputs: dict[str, str]
    dtype: str = "f32"
    min_n: int = 1
    max_n: int = 16777216
    alias: str = "disjoint"
    domain: str = "finite"
    rtol: float = 2e-4
    atol: float = 2e-5

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - cls.__dataclass_fields__.keys():
            raise ValueError("unknown or malformed contract fields")
        return cls(**data)


@dataclass
class DagAIR:
    nodes: tuple[DagNode, ...]
    contract: DagContract
    v_meta: VMeta = field(default_factory=VMeta)

    @classmethod
    def from_proposal(cls, proposal, contract):
        if (not isinstance(proposal, dict) or set(proposal) != {"pattern", "nodes"}
                or proposal["pattern"] not in ("dag", "compose")):
            raise ValueError("proposal requires only pattern=dag/compose and nodes")
        nodes = proposal["nodes"]
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 128:
            raise ValueError("graph requires 1..128 nodes")
        parsed = []
        for item in nodes:
            if not isinstance(item, dict) or set(item) - DagNode.__dataclass_fields__.keys():
                raise ValueError("unknown or malformed node fields")
            try:
                parsed.append(DagNode(**item))
            except TypeError as exc:
                raise ValueError("missing node fields") from exc
        graph = cls(tuple(parsed), contract)
        check_dag(graph)
        return graph

    def to_dict(self):
        return {"pattern": "dag", "nodes": [asdict(n) for n in self.nodes],
                "contract": asdict(self.contract), "v_meta": asdict(self.v_meta)}

    def fingerprint(self):
        import hashlib
        record = self.to_dict()
        record.pop("v_meta")
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CheckedGraph:
    order: tuple[DagNode, ...]
    expressions: dict[str, DagExpr]
    shapes: dict[str, str]


def check_dag(graph: DagAIR) -> CheckedGraph:
    """Check shape flow, SSA hazards, reachability and acyclicity afresh."""
    graph.v_meta = VMeta()
    c = graph.contract
    try:
        if c.dtype != "f32" or c.alias != "disjoint" or c.domain != "finite":
            raise ValueError("DAG backend requires f32, disjoint storage and finite inputs")
        if type(c.min_n) is not int or type(c.max_n) is not int or not 1 <= c.min_n <= c.max_n <= 16777216:
            raise ValueError("invalid positive n domain")
        for value in (c.rtol, c.atol):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("tolerances must be nonnegative and finite")
        for slots in (c.inputs, c.outputs):
            if not isinstance(slots, dict) or not slots or len(slots) > 32:
                raise ValueError("contract requires 1..32 named inputs and outputs")
            if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                   or name == "n" or shape not in ("n", "scalar") for name, shape in slots.items()):
                raise ValueError("invalid contract name or extent")
        if set(c.inputs) & set(c.outputs):
            raise ValueError("input/output alias is not admitted")
        if not 1 <= len(graph.nodes) <= 128:
            raise ValueError("graph requires 1..128 nodes")
        nodes, exprs = {}, {}
        for node in graph.nodes:
            if (not isinstance(node.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.name)
                    or node.name == "n" or node.name in nodes or node.name in c.inputs):
                raise ValueError("SSA write hazard: duplicate, invalid or overwritten input name")
            if node.pattern not in ("map", "reduce", "contract"):
                raise ValueError("unsupported DAG node pattern")
            if node.equiv not in ("bitexact", "tolerance"):
                raise ValueError("unknown node equivalence policy")
            if node.pattern != "map":
                if node.reduce_op not in ("sum", "max", "min"):
                    raise ValueError("unsupported contraction operation")
                if node.reduce_op == "sum" and node.equiv != "tolerance":
                    raise ValueError("FP sum reassociation requires tolerance")
            elif node.reduce_op is not None:
                raise ValueError("map cannot contain a reduction field")
            nodes[node.name], exprs[node.name] = node, expression(node.formula)
            if node.equiv == "bitexact":
                # A map copy is bit preserving; arithmetic needs a numerical policy.
                if node.pattern != "map" or exprs[node.name].op != "read":
                    raise ValueError("bitexact is supported only for direct map copies")
        if not set(c.outputs) <= nodes.keys():
            raise ValueError("contract output is not produced")
        available = set(c.inputs) | nodes.keys()
        for value in exprs.values():
            if value.reads() - available:
                raise ValueError("expression reads an undefined value")
        pending, order, shapes = dict(nodes), [], dict(c.inputs)
        while pending:
            ready = [node for node in pending.values() if exprs[node.name].reads() <= shapes.keys()]
            if not ready:
                raise ValueError("cyclic graph dependencies")
            for node in ready:
                shape = "n" if any(shapes[r] == "n" for r in exprs[node.name].reads()) else "scalar"
                if node.pattern != "map":
                    if shape != "n":
                        raise ValueError("reduction must consume a vector expression")
                    shape = "scalar"
                shapes[node.name] = shape
                order.append(node)
                del pending[node.name]
        if any(shapes[name] != shape for name, shape in c.outputs.items()):
            raise ValueError("output extent disagrees with the independent contract")
        live = set(c.outputs)
        for node in reversed(order):
            if node.name in live:
                live |= exprs[node.name].reads()
        if nodes.keys() - live:
            raise ValueError("dead nodes are not part of the output computation")
        graph.v_meta = VMeta(Status.CHECKED, "types, extents, SSA storage, dependencies and topology checked")
        return CheckedGraph(tuple(order), exprs, shapes)
    except (ValueError, TypeError, KeyError) as exc:
        graph.v_meta.proof = str(exc)
        raise ValueError(str(exc)) from exc
