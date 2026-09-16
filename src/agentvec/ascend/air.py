"""Agentic Intermediate Representation (AIR) for the cuBLAS -> Ascend migration.

Adapted from the AgentVec artifact (RISC-V VLA) to target the Ascend AI Core.
AIR is a FIXED, typed schema (not free LLM tokens): the Intent Proposer may only
fill declared slots. The four-tuple is

    I = < S_algo , C_phy , M_map , V_meta >

  S_algo : hardware-independent computational intent (what is computed + the
           rewrites its rules permit) — the BLAS pattern and its element op.
  C_phy  : immutable correctness constraints from static analysis — dtype,
           dependence, alias, and the FP-reassociation tier `equiv`.
  M_map  : memory-access regularity — access pattern, stride, matrix layout, bound.
  V_meta : verification lifecycle — DRAFT -> CHECKED -> VERIFIED (+ proof).

The static contract checks in guard.py establish CHECKED. VERIFIED is reserved
for the execution report after compilation and independent target checks.
Deserialization never imports a previous verification status.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import json


# ---- S_algo : the BLAS computational intent ---------------------------------
class Pattern(str, Enum):
    MAP = "map"        # elementwise: out[i] = f(in0[i], in1[i], ...)   (L1: axpy/scal/copy)
    REDUCE = "reduce"  # out[0] = post(fold(op, pre(in[i])))            (L1: dot/asum/nrm2)
    GEMV = "gemv"      # y = alpha*op(A)*x + beta*y                     (L2)
    GEMM = "gemm"      # C = alpha*A*B + beta*C    (NN)                 (L3)
    GEMM_NT = "gemm_nt"  # C = alpha*A*B^T + beta*C (NT, row-dot form)  (L3)
    SYRK = "syrk"      # C = alpha*A*A^T + beta*C  (symmetric rank-k)   (L3)
    TRSV = "trsv"      # triangular solve A*x = b  (loop-carried => VETO demo)
    TRSM = "trsm"      # triangular solve A*X = B  (multi-RHS, loop-carried => VETO)


# elem_op is a small typed expression AST over named inputs + scalar consts.
# kinds: load(src) | const(val) | add/sub/mul(a,b) | mulc(a, const) | abs(a) | fma(a,b,c)
@dataclass
class Expr:
    op: str
    src: Optional[str] = None
    val: Optional[float] = None
    args: list = field(default_factory=list)

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
    elem_op: Optional[Expr] = None        # MAP: per-element expression
    reduce_op: Optional[str] = None       # REDUCE: sum|max|min
    reduce_pre: Optional[str] = None      # REDUCE: a | ab | abs_a | aa
    reduce_post: Optional[str] = None     # REDUCE: None | sqrt
    trans_a: bool = False                 # GEMV/GEMM: transpose A
    trans_b: bool = False                 # GEMM: transpose B
    has_alpha: bool = True                # scalar alpha present
    has_beta: bool = True                 # scalar beta present


# ---- C_phy : immutable correctness constraints -----------------------------
class Dep(str, Enum):
    NONE = "none"
    LOOP_CARRIED = "loop_carried"
    REDUCTION = "reduction"


class Alias(str, Enum):
    DISJOINT = "disjoint"
    MAY_ALIAS = "may_alias"


@dataclass
class CPhy:
    dtype: str = "f32"           # f16 | f32 | i32 (Ascend AI Core: f16 cube-native, f32 vector)
    dep: Dep = Dep.NONE
    dep_distance: int = 0
    alias: Alias = Alias.DISJOINT
    equiv: str = "ulp"           # exact | ulp  (ulp licenses FP reassociation for reductions)


# ---- M_map : memory-access regularity --------------------------------------
class Access(str, Enum):
    UNIT_STRIDE = "unit_stride"
    STRIDED = "strided"
    INDEXED = "indexed"


class Layout(str, Enum):
    NA = "na"
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"     # cuBLAS native is column-major


@dataclass
class MMap:
    access: Access = Access.UNIT_STRIDE
    stride: int = 1
    layout: Layout = Layout.NA
    bound: str = "n"            # affine extent (kept symbolic): n / m,n / m,n,k


# ---- V_meta : verification lifecycle ---------------------------------------
class Status(str, Enum):
    DRAFT = "DRAFT"
    CHECKED = "CHECKED"
    VERIFIED = "VERIFIED"


@dataclass
class VMeta:
    status: Status = Status.DRAFT
    proof: str = ""


@dataclass
class AIR:
    s_algo: SAlgo
    c_phy: CPhy = field(default_factory=CPhy)
    m_map: MMap = field(default_factory=MMap)
    v_meta: VMeta = field(default_factory=VMeta)
    name: str = "op"            # operator label (e.g. saxpy)

    # ---- serialization ----
    def to_json(self) -> str:
        d = {
            "name": self.name,
            "s_algo": {
                "pattern": self.s_algo.pattern.value,
                "elem_op": self.s_algo.elem_op.to_dict() if self.s_algo.elem_op else None,
                "reduce_op": self.s_algo.reduce_op,
                "reduce_pre": self.s_algo.reduce_pre,
                "reduce_post": self.s_algo.reduce_post,
                "trans_a": self.s_algo.trans_a, "trans_b": self.s_algo.trans_b,
                "has_alpha": self.s_algo.has_alpha, "has_beta": self.s_algo.has_beta,
            },
            "c_phy": {"dtype": self.c_phy.dtype, "dep": self.c_phy.dep.value,
                      "dep_distance": self.c_phy.dep_distance, "alias": self.c_phy.alias.value,
                      "equiv": self.c_phy.equiv},
            "m_map": {"access": self.m_map.access.value, "stride": self.m_map.stride,
                      "layout": self.m_map.layout.value, "bound": self.m_map.bound},
            "v_meta": {"status": self.v_meta.status.value, "proof": self.v_meta.proof},
        }
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "AIR":
        d = json.loads(s)
        sa, cp, mm, vm = d["s_algo"], d["c_phy"], d.get("m_map", {}), d.get("v_meta", {})
        return AIR(
            name=d.get("name", "op"),
            s_algo=SAlgo(pattern=Pattern(sa["pattern"]), elem_op=Expr.from_dict(sa.get("elem_op")),
                         reduce_op=sa.get("reduce_op"), reduce_pre=sa.get("reduce_pre"),
                         reduce_post=sa.get("reduce_post"), trans_a=sa.get("trans_a", False),
                         trans_b=sa.get("trans_b", False), has_alpha=sa.get("has_alpha", True),
                         has_beta=sa.get("has_beta", True)),
            c_phy=CPhy(dtype=cp.get("dtype", "f32"), dep=Dep(cp.get("dep", "none")),
                       dep_distance=cp.get("dep_distance", 0), alias=Alias(cp.get("alias", "disjoint")),
                       equiv=cp.get("equiv", "ulp")),
            m_map=MMap(access=Access(mm.get("access", "unit_stride")), stride=mm.get("stride", 1),
                       layout=Layout(mm.get("layout", "na")), bound=mm.get("bound", "n")),
            v_meta=VMeta(),
        )

    # ---- structural validation (schema well-formedness, NOT correctness) ----
    def validate_schema(self) -> tuple[bool, str]:
        if self.c_phy.dtype not in ("f16", "f32", "i32"):
            return False, f"unsupported dtype {self.c_phy.dtype}"
        p = self.s_algo.pattern
        if p == Pattern.MAP and self.s_algo.elem_op is None:
            return False, "MAP requires elem_op"
        if p == Pattern.REDUCE:
            if self.s_algo.reduce_op not in ("sum", "max", "min"):
                return False, f"REDUCE needs reduce_op in sum/max/min, got {self.s_algo.reduce_op}"
            if self.s_algo.reduce_pre not in ("a", "ab", "abs_a", "aa"):
                return False, f"REDUCE pre {self.s_algo.reduce_pre} unsupported"
        if p in (Pattern.GEMV, Pattern.GEMM, Pattern.GEMM_NT, Pattern.SYRK) and self.m_map.layout == Layout.NA:
            return False, f"{p.value} requires a matrix layout"
        return True, "well-formed"
