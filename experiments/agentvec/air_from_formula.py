"""Parse a high-level intent formula (what the LLM recovers) into an AIR Expr AST.

The AgentVec arm asks the model only for the ALGORITHMIC INTENT, e.g.
  {"pattern":"map","dtype":"f32","formula":"out = 2*a + b"}
  {"pattern":"reduce","dtype":"f32","reduce_op":"sum","formula":"a*b"}   (dot)
  {"pattern":"reduce","dtype":"f32","reduce_op":"sum","formula":"abs(a)"} (asum)
This is far easier (and more robust across models) than emitting correct RVV; the
deterministic Guard+lowering then guarantee a correct, VLA-scalable kernel.

Supported map grammar: + - * , scalar*array (mulc), array*array (mul), abs(), over
inputs a,b,c and numeric constants.
"""
import re
from air import AIR, SAlgo, CPhy, MMap, Expr, Pattern, Dep, Alias, Access


class _P:
    def __init__(self, s):
        self.toks = re.findall(r"\d+\.?\d*|[A-Za-z_]\w*|[+\-*()]", s)
        self.i = 0
    def peek(self): return self.toks[self.i] if self.i < len(self.toks) else None
    def next(self): t = self.peek(); self.i += 1; return t

    def expr(self):
        e = self.term()
        while self.peek() in ("+", "-"):
            op = self.next(); r = self.term()
            e = Expr("add" if op == "+" else "sub", args=[e, r])
        return e
    def term(self):
        e = self.factor()
        while self.peek() == "*":
            self.next(); r = self.factor()
            # scalar*array -> mulc ; else mul
            if e.op == "const":
                e = Expr("mulc", val=e.val, args=[r])
            elif r.op == "const":
                e = Expr("mulc", val=r.val, args=[e])
            else:
                e = Expr("mul", args=[e, r])
        return e
    def factor(self):
        t = self.next()
        if t == "(":
            e = self.expr(); assert self.next() == ")"; return e
        if t == "abs":
            assert self.next() == "("; e = self.expr(); assert self.next() == ")"
            return Expr("abs", args=[e])
        if re.match(r"^\d", t):
            return Expr("const", val=float(t))
        return Expr("load", src=t)        # identifier => array load


def parse_formula(formula: str) -> Expr:
    rhs = formula.split("=", 1)[1] if "=" in formula else formula
    return _P(rhs.strip()).expr()


def air_from_intent(intent: dict) -> AIR:
    """Build an AIR from the model's recovered intent dict."""
    dtype = intent.get("dtype", "f32")
    pat = intent["pattern"]
    if pat == "map":
        return AIR(s_algo=SAlgo(pattern=Pattern.MAP, elem_op=parse_formula(intent["formula"])),
                   c_phy=CPhy(dtype=dtype, dep=Dep.NONE, alias=Alias.DISJOINT),
                   m_map=MMap(access=Access.UNIT_STRIDE))
    elif pat == "reduce":
        return AIR(s_algo=SAlgo(pattern=Pattern.REDUCE, reduce_op=intent.get("reduce_op", "sum"),
                                reduce_src=intent.get("reduce_src", "a")),
                   c_phy=CPhy(dtype=dtype, dep=Dep.REDUCTION, alias=Alias.DISJOINT))
    raise ValueError(f"unsupported pattern {pat}")


if __name__ == "__main__":
    for f in ["out = 2*a + b", "3*a - b", "a*b"]:
        print(f, "->", parse_formula(f).to_dict())
