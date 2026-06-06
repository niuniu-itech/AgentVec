"""Agentic Intermediate Representation (AIR).

AIR is a FIXED, typed schema (not free LLM tokens): the Intent Proposer may only
fill declared slots. This module defines the four-tuple

    I = < S_algo , C_phy , M_map , V_meta >

as dataclasses with JSON (de)serialization, plus a structural validator. The
deductive checks that flip V_meta to VERIFIED live in guard.py; here we only
enforce that an instance is well-formed against the schema.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union
import json


# ---- S_algo : hardware-independent computational intent -------------------
class Pattern(str, Enum):
    MAP = "map"            # elementwise out[i] = f(in0[i], in1[i], ...)
    REDUCE = "reduce"      # out[0] = fold(op, in0[i])
    SCAN = "scan"          # prefix scan (reserved)
    STENCIL = "stencil"    # out[i] = f(in[i-k..i+k]) (reserved)
    GATHER_APPLY = "gather_apply"  # indexed load then map (reserved)


# elem_op is a small typed expression AST over named inputs + scalar consts.
# Node kinds: load(src), const(val), add/sub/mul(a,b), mulc(a,const), fma(a,b,c)=a*b+c
@dataclass
class Expr:
    op: str                       # load|const|add|sub|mul|mulc|fma
    src: Optional[str] = None     # for load: input array name
    val: Optional[float] = None   # for const / mulc scalar
    args: list = field(default_factory=list)  # child Expr for binary/ternary ops

    @staticmethod
    def from_dict(d: Union[dict, None]) -> Optional["Expr"]:
        if d is None:
            return None
        return Expr(op=d["op"], src=d.get("src"), val=d.get("val"),
                    args=[Expr.from_dict(a) for a in d.get("args", [])])

    def to_dict(self) -> dict:
        out = {"op": self.op}
        if self.src is not None: out["src"] = self.src
        if self.val is not None: out["val"] = self.val
        if self.args: out["args"] = [a.to_dict() for a in self.args]
        return out

    def inputs(self) -> set:
        s = {self.src} if self.op == "load" and self.src else set()
        for a in self.args:
            s |= a.inputs()
        return s


@dataclass
class SAlgo:
    pattern: Pattern
    elem_op: Optional[Expr] = None     # for MAP: per-element expression
    reduce_op: Optional[str] = None    # for REDUCE: sum|max|min
    reduce_src: Optional[str] = None   # for REDUCE: input array name


# ---- C_phy : immutable correctness constraints (from static analysis) ------
class Dep(str, Enum):
    NONE = "none"
    LOOP_CARRIED = "loop_carried"
    REDUCTION = "reduction"


class Alias(str, Enum):
    DISJOINT = "disjoint"
    MAY_ALIAS = "may_alias"


@dataclass
class CPhy:
    dtype: str                 # f32 | i32
    dep: Dep = Dep.NONE
    dep_distance: int = 0      # for LOOP_CARRIED
    alias: Alias = Alias.DISJOINT


# ---- M_map : memory-access regularity --------------------------------------
class Access(str, Enum):
    UNIT_STRIDE = "unit_stride"
    STRIDED = "strided"
    INDEXED = "indexed"


@dataclass
class MMap:
    access: Access = Access.UNIT_STRIDE
    stride: int = 1            # for STRIDED
    bound: str = "n"           # affine expr in loop iv (kept symbolic)


# ---- V_meta : verification lifecycle ---------------------------------------
class Status(str, Enum):
    DRAFT = "DRAFT"
    CHECKED = "CHECKED"
    VERIFIED = "VERIFIED"


@dataclass
class VMeta:
    status: Status = Status.DRAFT
    proof: str = ""            # dependence-test witness or rejection reason


@dataclass
class AIR:
    s_algo: SAlgo
    c_phy: CPhy
    m_map: MMap = field(default_factory=MMap)
    v_meta: VMeta = field(default_factory=VMeta)

    # ---- serialization ----
    def to_json(self) -> str:
        d = {
            "s_algo": {"pattern": self.s_algo.pattern.value,
                       "elem_op": self.s_algo.elem_op.to_dict() if self.s_algo.elem_op else None,
                       "reduce_op": self.s_algo.reduce_op,
                       "reduce_src": self.s_algo.reduce_src},
            "c_phy": {"dtype": self.c_phy.dtype, "dep": self.c_phy.dep.value,
                      "dep_distance": self.c_phy.dep_distance, "alias": self.c_phy.alias.value},
            "m_map": {"access": self.m_map.access.value, "stride": self.m_map.stride, "bound": self.m_map.bound},
            "v_meta": {"status": self.v_meta.status.value, "proof": self.v_meta.proof},
        }
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "AIR":
        d = json.loads(s)
        sa = d["s_algo"]
        cp = d["c_phy"]
        mm = d.get("m_map", {})
        vm = d.get("v_meta", {})
        return AIR(
            s_algo=SAlgo(pattern=Pattern(sa["pattern"]),
                         elem_op=Expr.from_dict(sa.get("elem_op")),
                         reduce_op=sa.get("reduce_op"), reduce_src=sa.get("reduce_src")),
            c_phy=CPhy(dtype=cp["dtype"], dep=Dep(cp.get("dep", "none")),
                       dep_distance=cp.get("dep_distance", 0), alias=Alias(cp.get("alias", "disjoint"))),
            m_map=MMap(access=Access(mm.get("access", "unit_stride")),
                       stride=mm.get("stride", 1), bound=mm.get("bound", "n")),
            v_meta=VMeta(status=Status(vm.get("status", "DRAFT")), proof=vm.get("proof", "")),
        )

    # ---- structural validation (schema well-formedness, NOT correctness) ----
    def validate_schema(self) -> tuple[bool, str]:
        if self.c_phy.dtype not in ("f32", "i32"):
            return False, f"unsupported dtype {self.c_phy.dtype}"
        if self.s_algo.pattern == Pattern.MAP:
            if self.s_algo.elem_op is None:
                return False, "MAP requires elem_op"
        elif self.s_algo.pattern == Pattern.REDUCE:
            if self.s_algo.reduce_op not in ("sum", "max", "min"):
                return False, f"REDUCE needs reduce_op in sum/max/min, got {self.s_algo.reduce_op}"
            if not self.s_algo.reduce_src:
                return False, "REDUCE requires reduce_src"
        else:
            return False, f"pattern {self.s_algo.pattern} not yet supported by lowering"
        return True, "well-formed"


if __name__ == "__main__":
    # hand-authored AIR for saxpy: out[i] = 2*a[i] + b[i]
    saxpy = AIR(
        s_algo=SAlgo(pattern=Pattern.MAP,
                     elem_op=Expr("add", args=[
                         Expr("mulc", val=2.0, args=[Expr("load", src="a")]),
                         Expr("load", src="b")])),
        c_phy=CPhy(dtype="f32", dep=Dep.NONE, alias=Alias.DISJOINT),
        m_map=MMap(access=Access.UNIT_STRIDE, bound="n"),
    )
    ok, msg = saxpy.validate_schema()
    print("schema:", ok, msg)
    print(saxpy.to_json())
    # round-trip
    assert AIR.from_json(saxpy.to_json()).to_json() == saxpy.to_json()
    print("round-trip OK")
