"""Parse bounded intent expressions without executing model-supplied text."""
import ast
from .air import AIR, SAlgo, CPhy, Expr, Pattern, Dep


def parse_formula(formula: str) -> Expr:
    if not isinstance(formula, str) or not formula.strip() or len(formula) > 4096:
        raise ValueError("formula must be a nonempty string of at most 4096 characters")
    try:
        module = ast.parse(formula.strip(), mode="exec")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid formula syntax") from exc
    if len(module.body) != 1:
        raise ValueError("expected one expression or out assignment")
    statement = module.body[0]
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name) or statement.targets[0].id != "out":
            raise ValueError("only assignment to out is supported")
        node = statement.value
    elif isinstance(statement, ast.Expr):
        node = statement.value
    else:
        raise ValueError("expected an arithmetic expression")

    def convert(n, depth=0):
        if depth > 32:
            raise ValueError("expression exceeds depth 32")
        if isinstance(n, ast.Name) and n.id in ("a", "b", "c"):
            return Expr("load", src=n.id)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Expr("const", val=n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            value = convert(n.operand, depth + 1)
            if isinstance(n.op, ast.UAdd):
                return value
            if value.op == "const":
                return Expr("const", val=-value.val)
            return Expr("mulc", val=-1, args=[value])
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = convert(n.left, depth + 1), convert(n.right, depth + 1)
            if isinstance(n.op, ast.Mult):
                if left.op == "const":
                    return Expr("mulc", val=left.val, args=[right])
                if right.op == "const":
                    return Expr("mulc", val=right.val, args=[left])
            return Expr({ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}[type(n.op)], args=[left, right])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "abs" and len(n.args) == 1 and not n.keywords:
            return Expr("abs", args=[convert(n.args[0], depth + 1)])
        raise ValueError("unsupported formula; use a, b, c, constants, +, -, *, or abs")

    return convert(node)


def air_from_intent(intent: dict, *, constraints: CPhy | None = None) -> AIR:
    """Attach caller-owned rules; the proposer cannot supply legality facts.

    Without constraints, assume the registered disjoint, unit-stride harness ABI.
    This helper does not infer dependence or aliasing from arbitrary source.
    """
    if not isinstance(intent, dict):
        raise ValueError("intent must be a JSON object")
    pattern = Pattern(intent["pattern"])
    dtype = intent["dtype"]
    if constraints is not None and dtype != constraints.dtype:
        raise ValueError("proposed dtype disagrees with the source contract")
    rules = constraints or CPhy(dtype, dep=Dep.REDUCTION if pattern == Pattern.REDUCE else Dep.NONE)
    if pattern == Pattern.MAP:
        allowed = {"pattern", "dtype", "formula"}
        algo = SAlgo(pattern, elem_op=parse_formula(intent["formula"]))
    elif pattern == Pattern.REDUCE:
        allowed = {"pattern", "dtype", "reduce_op", "elem", "formula", "reduce_src", "postproc"}
        expressions = [intent[key] for key in ("elem", "formula", "reduce_src") if key in intent]
        if len(expressions) != 1:
            raise ValueError("reduction needs exactly one element expression")
        post = intent.get("postproc")
        algo = SAlgo(pattern, elem_op=parse_formula(expressions[0]), reduce_op=intent["reduce_op"],
                     postproc=None if post in (None, "none") else post)
    else:
        raise ValueError(f"no core lowering for {pattern.value}")
    if set(intent) - allowed:
        raise ValueError("intent contains unknown fields")
    air = AIR(algo, rules)
    ok, message = air.validate_schema()
    if not ok:
        raise ValueError(message)
    return air
