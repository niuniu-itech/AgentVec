"""Phase 3: deterministic lowering of a VERIFIED AIR to RISC-V VLA C.

No probabilistic reasoning here. Given an AIR instance, this mechanically emits
a strip-mined, vector-length-agnostic RVV 1.0 kernel (Policy B: scalable control
via vsetvl) whose body realizes S_algo under C_phy (Policy A: semantics->intrinsics).
Output conforms to the difftest harness contract:
    #define DT / DT_IS_FLOAT / REDUCE, agentvec_kernel(...), #include "harness.h"
so the same independent differential tester validates correctness.
"""
from air import AIR, Pattern, Expr

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
    return (f"#include <riscv_vector.h>\n#include <stdint.h>\n"
            f"#define DT {dt}\n#define DT_IS_FLOAT {1 if is_float else 0}\n#define REDUCE {reduce}\n")


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
        c = f"{e.val}f" if is_float else f"{int(e.val)}"
        lines.append(f"        {m['vt']} {v} = {m['splat']}({c}, vl);")
        return v
    if e.op == "mulc":
        a = _emit_expr(e.args[0], m, is_float, lines, ctr)
        v = fresh()
        c = f"{e.val}f" if is_float else f"{int(e.val)}"
        lines.append(f"        {m['vt']} {v} = {m['mulc']}({a}, {c}, vl);")
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
        src = air.s_algo.reduce_src
        init = m["zero"]
        body.append(f"    size_t vlmax = {m['setvlmax']}();")
        body.append(f"    {m['vt']} acc = {m['splat']}({init}, vlmax);")
        body.append("    size_t vl;")
        body.append("    for (int i = 0; i < n; i += (int)vl) {")
        body.append(f"        vl = {m['setvl']}((size_t)(n - i));")
        body.append(f"        {m['vt']} v = {m['vle']}({src} + i, vl);")
        body.append(f"        acc = {red}(v, acc, vl);")
        body.append("    }")
        body.append(f"    {m['vse']}(out, acc, 1);")
    body.append("}")
    body.append('#include "harness.h"')
    return "\n".join(body) + "\n"


def emit_scalar_kernel(air: AIR) -> str:
    """Trivial scalar reference emitter (oracle), from the same AIR."""
    is_float = air.c_phy.dtype == "f32"
    dt = "float" if is_float else "int32_t"
    reduce = 1 if air.s_algo.pattern == Pattern.REDUCE else 0
    out = [f"#include <stdint.h>\n#define DT {dt}\n#define DT_IS_FLOAT {1 if is_float else 0}\n#define REDUCE {reduce}\n"]
    out.append("void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {")
    out.append("    (void)a; (void)b; (void)c;")

    def sexpr(e: Expr) -> str:
        if e.op == "load": return f"{e.src}[i]"
        if e.op == "const": return (f"{e.val}f" if is_float else f"{int(e.val)}")
        if e.op == "mulc":
            c = f"{e.val}f" if is_float else f"{int(e.val)}"
            return f"({c} * {sexpr(e.args[0])})"
        if e.op in ("add", "sub", "mul"):
            o = {"add": "+", "sub": "-", "mul": "*"}[e.op]
            return f"({sexpr(e.args[0])} {o} {sexpr(e.args[1])})"
        if e.op == "fma":
            return f"(({sexpr(e.args[0])} * {sexpr(e.args[1])}) + {sexpr(e.args[2])})"
        raise ValueError(e.op)

    if air.s_algo.pattern == Pattern.MAP:
        out.append("    for (int i = 0; i < n; i++) out[i] = " + sexpr(air.s_algo.elem_op) + ";")
    else:
        op = air.s_algo.reduce_op; src = air.s_algo.reduce_src
        if op == "sum":
            init = "0.0f" if is_float else "0"
            out.append(f"    DT s = {init}; for (int i = 0; i < n; i++) s += {src}[i]; out[0] = s;")
        else:
            cmp = ">" if op == "max" else "<"
            out.append(f"    DT s = {src}[0]; for (int i = 1; i < n; i++) if ({src}[i] {cmp} s) s = {src}[i]; out[0] = s;")
    out.append("}")
    out.append('#include "harness.h"')
    return "\n".join(out) + "\n"


def emit_reduce_kernel(reduce_op: str, pre_kind: str, postproc=None) -> str:
    """difftest-format f32 reduction: out[0] = reduce_op over pre(a,b), opt sqrt.
    pre_kind in {a, ab, abs_a, aa}. reduce_op in {sum, max, min}."""
    m = F
    pre = {
        "a":     f"{m['vle']}(a + i, vl)",
        "ab":    f"{m['mul']}({m['vle']}(a + i, vl), {m['vle']}(b + i, vl), vl)",
        "abs_a": f"__riscv_vfabs_v_f32m1({m['vle']}(a + i, vl), vl)",
        "aa":    f"__riscv_vfmul_vv_f32m1({m['vle']}(a + i, vl), {m['vle']}(a + i, vl), vl)",
    }[pre_kind]
    comb = {"sum": m["add"], "max": "__riscv_vfmax_vv_f32m1", "min": "__riscv_vfmin_vv_f32m1"}[reduce_op]
    red = {"sum": m["redsum"], "max": m["redmax"], "min": m["redmin"]}[reduce_op]
    init = {"sum": "0.0f", "max": "-3.0e38f", "min": "3.0e38f"}[reduce_op]
    out = ["#include <riscv_vector.h>\n#include <math.h>\n#include <stdint.h>\n"
           "#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"]
    out.append("void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {")
    out.append("    (void)b; (void)c;")
    out.append(f"    size_t vlmax = {m['setvlmax']}();")
    out.append(f"    {m['vt']} acc = {m['splat']}({init}, vlmax);")
    out.append("    size_t vl;")
    out.append("    for (int i = 0; i < n; i += (int)vl) {")
    out.append(f"        vl = {m['setvl']}((size_t)(n - i));")
    out.append(f"        {m['vt']} v = {pre};")
    out.append(f"        acc = {comb}(acc, v, vl);")
    out.append("    }")
    out.append(f"    {m['vt']} r = {m['splat']}({init}, 1);")
    out.append(f"    r = {red}(acc, r, vlmax);")
    out.append(f"    {m['vse']}(out, r, 1);")
    if postproc == "sqrt":
        out.append("    out[0] = sqrtf(out[0]);")
    out.append("}")
    out.append('#include "harness.h"')
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    from air import SAlgo, CPhy, MMap, Dep, Alias, Access, Pattern as P
    saxpy = AIR(s_algo=SAlgo(pattern=P.MAP,
                elem_op=Expr("add", args=[Expr("mulc", val=2.0, args=[Expr("load", src="a")]),
                                          Expr("load", src="b")])),
                c_phy=CPhy(dtype="f32"))
    print(emit_rvv_kernel(saxpy))
