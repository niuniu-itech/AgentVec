"""Attach independently registered rules to a bounded Ascend intent proposal."""
from dataclasses import fields
import json

from .air import AIR, Expr, SAlgo
from .registry import CUBLAS_SRC, recover


def from_proposal(op, proposal):
    if not isinstance(proposal, dict) or set(proposal) != {"s_algo"}:
        raise ValueError("Ascend proposal must contain only s_algo; rules and status are caller-owned")
    algorithm = proposal["s_algo"]
    if not isinstance(algorithm, dict) or set(algorithm) - {f.name for f in fields(SAlgo)}:
        raise ValueError("unknown algorithm fields")
    if "pattern" not in algorithm:
        raise ValueError("missing pattern")
    for field in ("trans_a", "trans_b", "has_alpha", "has_beta"):
        if field in algorithm and type(algorithm[field]) is not bool:
            raise ValueError("algorithm flags must be booleans")

    def check_expr(expr, depth=0):
        if expr is None:
            return
        if (depth > 32 or not isinstance(expr, dict) or 'op' not in expr
                or set(expr) - {'op','src','val','args'}):
            raise ValueError("malformed element expression")
        if not isinstance(expr.get('args',[]),list):
            raise ValueError("expression args must be an array")
        for child in expr.get('args',[]):
            check_expr(child,depth+1)
    check_expr(algorithm.get('elem_op'))
    registered = json.loads(recover(op).to_json())
    registered['s_algo'] = algorithm
    registered.pop('v_meta')
    return AIR.from_json(json.dumps(registered))


def propose(op, model):
    from ..llm_backend import chat, extract_json
    prompt = '''Recover the computational intent of this registered cuBLAS source.
Return only JSON {"s_algo": {...}}. Allowed fields: pattern, elem_op, reduce_op,
reduce_pre, reduce_post, trans_a, trans_b, has_alpha, has_beta.
Patterns: map, reduce, gemv, gemm, gemm_nt, syrk, trsv, trsm.
Expressions: {"op":"load","src":"x" or "y"},
{"op":"mulc","val":null,"args":[expr]} for runtime alpha multiplication,
{"op":"add","args":[expr,expr]} for elementwise addition.
Reduction pre: ab (x*y), abs_a (abs(x)), aa (x*x); reduce_op=sum;
reduce_post=sqrt only for nrm2, null otherwise. Set scalar flags and transpose flags
according to the source. Pure reductions have has_alpha=true,has_beta=true as
reserved schema flags. All map contracts have has_beta=false; copy has_alpha=false.
Do not output names, schedules, rules, status, proofs, or source code.
Source: ''' + CUBLAS_SRC[op]
    response = chat(model,prompt,max_tokens=1600)
    return from_proposal(op,extract_json(response)), response
