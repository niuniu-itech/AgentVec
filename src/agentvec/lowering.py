"""Deterministic lowering of contract-checked AIR to RVV C.

Generated candidates use runtime vsetvl and retain reduction tail lanes.
Independent differential testing is required before recording VERIFIED.
"""
from .air import AIR, Pattern, Expr
from .guard import guard

F = {  # f32 intrinsic fragments
    "vt": "vfloat32m1_t", "vle": "__riscv_vle32_v_f32m1", "vse": "__riscv_vse32_v_f32m1",
    "setvl": "__riscv_vsetvl_e32m1", "setvlmax": "__riscv_vsetvlmax_e32m1",
    "add": "__riscv_vfadd_vv_f32m1", "sub": "__riscv_vfsub_vv_f32m1", "mul": "__riscv_vfmul_vv_f32m1",
    "mulc": "__riscv_vfmul_vf_f32m1", "splat": "__riscv_vfmv_v_f_f32m1",
    "redsum": "__riscv_vfredusum_vs_f32m1_f32m1", "redmax": "__riscv_vfredmax_vs_f32m1_f32m1",
    "redmin": "__riscv_vfredmin_vs_f32m1_f32m1", "zero": "0.0f",
}
I = {  # i32 intrinsic fragments
    "vt": "vint32m1_t", "vle": "__riscv_vle32_v_i32m1", "vse": "__riscv_vse32_v_i32m1",
    "setvl": "__riscv_vsetvl_e32m1", "setvlmax": "__riscv_vsetvlmax_e32m1",
    "add": "__riscv_vadd_vv_i32m1", "sub": "__riscv_vsub_vv_i32m1", "mul": "__riscv_vmul_vv_i32m1",
    "mulc": "__riscv_vmul_vx_i32m1", "splat": "__riscv_vmv_v_x_i32m1",
    "redsum": "__riscv_vredsum_vs_i32m1_i32m1", "redmax": "__riscv_vredmax_vs_i32m1_i32m1",
    "redmin": "__riscv_vredmin_vs_i32m1_i32m1", "zero": "0",
}


def _macros(air: AIR) -> str:
    is_float = air.c_phy.dtype == "f32"
    dt = "float" if is_float else "int32_t"
    reduce = 1 if air.s_algo.pattern == Pattern.REDUCE else 0
    return (f"#include <riscv_vector.h>\n#include <stdint.h>\n#include <math.h>\n#include <limits.h>\n"
            f"#define DT {dt}\n#define DT_IS_FLOAT {1 if is_float else 0}\n#define REDUCE {reduce}\n")


def _constant(value, is_float):
    if not is_float:
        return str(int(value))
    literal = format(float(value), ".9g")
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return literal + "f"


def _identity(op, is_float):
    return ({"sum": "0.0f", "min": "INFINITY", "max": "-INFINITY"} if is_float else
            {"sum": "0", "min": "INT32_MAX", "max": "INT32_MIN"})[op]


def _emit_expr(e: Expr, m: dict, is_float: bool, lines: list, ctr: list) -> str:
    """Emit code for expression e; return name of vreg holding its value."""
    def fresh():
        ctr[0] += 1
        return f"v{ctr[0]}"
    if e.op == "load":
        v = fresh()
        lines.append(f"        {m['vt']} {v} = {m['vle']}({e.src} + i, vl);")
        return v
    if e.op == "const":
        v = fresh()
        c = _constant(e.val, is_float)
        lines.append(f"        {m['vt']} {v} = {m['splat']}({c}, vl);")
        return v
    if e.op == "mulc":
        a = _emit_expr(e.args[0], m, is_float, lines, ctr)
        v = fresh()
        c = _constant(e.val, is_float)
        lines.append(f"        {m['vt']} {v} = {m['mulc']}({a}, {c}, vl);")
        return v
    if e.op == "abs":
        a = _emit_expr(e.args[0], m, is_float, lines, ctr)
        v = fresh()
        lines.append(f"        {m['vt']} {v} = __riscv_vfabs_v_f32m1({a}, vl);")
        return v
    if e.op in ("add", "sub", "mul"):
        a = _emit_expr(e.args[0], m, is_float, lines, ctr)
        b = _emit_expr(e.args[1], m, is_float, lines, ctr)
        v = fresh()
        lines.append(f"        {m['vt']} {v} = {m[e.op]}({a}, {b}, vl);")
        return v
    if e.op == "fma":  # a*b + c
        a = _emit_expr(e.args[0], m, is_float, lines, ctr)
        b = _emit_expr(e.args[1], m, is_float, lines, ctr)
        c = _emit_expr(e.args[2], m, is_float, lines, ctr)
        ab = fresh(); lines.append(f"        {m['vt']} {ab} = {m['mul']}({a}, {b}, vl);")
        v = fresh(); lines.append(f"        {m['vt']} {v} = {m['add']}({ab}, {c}, vl);")
        return v
    raise ValueError(f"unsupported expr op: {e.op}")


def emit_rvv_kernel(air: AIR) -> str:
    ok, reason = guard(air)
    if not ok:
        raise ValueError(f"guard veto: {reason}")
    is_float = air.c_phy.dtype == "f32"
    m = F if is_float else I
    body = [_macros(air)]
    body.append("void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {")
    body.append("    (void)a; (void)b; (void)c;")
    if air.s_algo.pattern == Pattern.MAP:
        body.append("    size_t vl;")
        body.append("    for (int i = 0; i < n; i += (int)vl) {")
        body.append(f"        vl = {m['setvl']}((size_t)(n - i));")
        lines = []; ctr = [-1]
        res = _emit_expr(air.s_algo.elem_op, m, is_float, lines, ctr)
        body.extend(lines)
        body.append(f"        {m['vse']}(out + i, {res}, vl);")
        body.append("    }")
    elif air.s_algo.pattern == Pattern.REDUCE:
        red = {"sum": m["redsum"], "max": m["redmax"], "min": m["redmin"]}[air.s_algo.reduce_op]
        init = _identity(air.s_algo.reduce_op, is_float)
        combine = ({"sum": "__riscv_vfadd_vv_f32m1_tu", "max": "__riscv_vfmax_vv_f32m1_tu",
                    "min": "__riscv_vfmin_vv_f32m1_tu"} if is_float else
                   {"sum": "__riscv_vadd_vv_i32m1_tu", "max": "__riscv_vmax_vv_i32m1_tu",
                    "min": "__riscv_vmin_vv_i32m1_tu"})[air.s_algo.reduce_op]
        body.append(f"    size_t vlmax = {m['setvlmax']}();")
        body.append(f"    {m['vt']} acc = {m['splat']}({init}, vlmax);")
        body.append("    size_t vl;")
        body.append("    for (int i = 0; i < n; i += (int)vl) {")
        body.append(f"        vl = {m['setvl']}((size_t)(n - i));")
        expression = air.s_algo.elem_op or Expr("load", src=air.s_algo.reduce_src)
        lines = []
        value = _emit_expr(expression, m, is_float, lines, [-1])
        body.extend(lines)
        # Preserve accumulated lanes when the last strip is shorter than VLMAX.
        body.append(f"        acc = {combine}(acc, acc, {value}, vl);")
        body.append("    }")
        body.append(f"    {m['vt']} seed = {m['splat']}({init}, 1);")
        body.append(f"    seed = {red}(acc, seed, vlmax);")
        body.append(f"    {m['vse']}(out, seed, 1);")
        if air.s_algo.postproc == "sqrt":
            body.append("    out[0] = sqrtf(out[0]);")
    body.append("}")
    body.append('#include "harness.h"')
    return "\n".join(body) + "\n"


def emit_scalar_kernel(air: AIR) -> str:
    """Scalar interpreter for lowering tests, not an independent intent oracle."""
    ok, reason = guard(air)
    if not ok:
        raise ValueError(f"guard veto: {reason}")
    is_float = air.c_phy.dtype == "f32"
    dt = "float" if is_float else "int32_t"
    reduce = 1 if air.s_algo.pattern == Pattern.REDUCE else 0
    out = [f"#include <stdint.h>\n#include <math.h>\n#include <limits.h>\n#include <string.h>\n#define DT {dt}\n#define DT_IS_FLOAT {1 if is_float else 0}\n#define REDUCE {reduce}\n"]
    if not is_float:
        out.append("static int32_t bits(uint32_t x) { int32_t y; memcpy(&y, &x, 4); return y; }")
    out.append("void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {")
    out.append("    (void)a; (void)b; (void)c;")

    def sexpr(e: Expr) -> str:
        if e.op == "load": return f"{e.src}[i]"
        if e.op == "const": return _constant(e.val, is_float)
        if e.op == "abs": return f"fabsf({sexpr(e.args[0])})"
        if e.op == "mulc":
            c = _constant(e.val, is_float)
            return f"({c} * {sexpr(e.args[0])})" if is_float else f"bits((uint32_t)({c}) * (uint32_t)({sexpr(e.args[0])}))"
        if e.op in ("add", "sub", "mul"):
            o = {"add": "+", "sub": "-", "mul": "*"}[e.op]
            if is_float:
                return f"({sexpr(e.args[0])} {o} {sexpr(e.args[1])})"
            return f"bits((uint32_t)({sexpr(e.args[0])}) {o} (uint32_t)({sexpr(e.args[1])}))"
        if e.op == "fma":
            return sexpr(Expr("add", args=[Expr("mul", args=e.args[:2]), e.args[2]]))
        raise ValueError(e.op)

    if air.s_algo.pattern == Pattern.MAP:
        out.append("    for (int i = 0; i < n; i++) out[i] = " + sexpr(air.s_algo.elem_op) + ";")
    else:
        op = air.s_algo.reduce_op
        value = sexpr(air.s_algo.elem_op or Expr("load", src=air.s_algo.reduce_src))
        init = _identity(op, is_float)
        if op == "sum":
            combine = f"s + ({value})" if is_float else f"bits((uint32_t)s + (uint32_t)({value}))"
            out.append(f"    DT s = {init}; for (int i = 0; i < n; i++) s = {combine}; out[0] = s;")
        else:
            cmp = ">" if op == "max" else "<"
            out.append(f"    DT s = {init}; for (int i = 0; i < n; i++) {{ DT v = {value}; if (v {cmp} s) s = v; }} out[0] = s;")
        if air.s_algo.postproc == "sqrt":
            out.append("    out[0] = sqrtf(out[0]);")
    out.append("}")
    out.append('#include "harness.h"')
    return "\n".join(out) + "\n"


def emit_reduce_kernel(reduce_op: str, pre_kind: str, postproc=None) -> str:
    """Compatibility entry: build and check the same AIR as the map/reduce path."""
    from .air_from_formula import air_from_intent
    expressions = {"a": "a", "ab": "a*b", "abs_a": "abs(a)", "aa": "a*a"}
    if pre_kind not in expressions:
        raise ValueError("unsupported reduction element expression")
    air = air_from_intent({"pattern": "reduce", "dtype": "f32", "reduce_op": reduce_op,
                           "elem": expressions[pre_kind], "postproc": postproc})
    return emit_rvv_kernel(air)
