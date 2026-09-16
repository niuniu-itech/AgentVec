"""Topological C/RVV lowering for checked map/reduction graphs."""
from dataclasses import dataclass, replace

from .dag import DagAIR, DagExpr, check_dag


@dataclass(frozen=True)
class DagSchedule:
    fuse_maps: bool = False


def plan(graph: DagAIR, schedule: DagSchedule):
    checked = check_dag(graph)
    if type(schedule.fuse_maps) is not bool:
        raise ValueError("fuse_maps must be boolean")
    nodes = list(checked.order)
    exprs = dict(checked.expressions)
    fused = []

    def count(expr, name):
        return int(expr.op == "read" and expr.name == name) + sum(count(a, name) for a in expr.args)

    def substitute(expr, name, value):
        return value if expr.op == "read" and expr.name == name else replace(
            expr, args=tuple(substitute(a, name, value) for a in expr.args))

    if schedule.fuse_maps:
        for producer in tuple(nodes):
            users = [node for node in nodes if producer.name in exprs[node.name].reads()]
            if (producer.pattern != "map" or producer.name in graph.contract.outputs
                    or checked.shapes[producer.name] != "n" or len(users) != 1):
                continue
            consumer = users[0]
            if count(exprs[consumer.name], producer.name) != 1:
                continue
            exprs[consumer.name] = substitute(exprs[consumer.name], producer.name, exprs[producer.name])
            nodes.remove(producer)
            fused.append([producer.name, consumer.name])
    return nodes, exprs, checked.shapes, fused


def emit_dag(graph: DagAIR, target="rvv", schedule=DagSchedule()):
    """Return source and a plan record; compilation/testing remains separate."""
    if target not in ("rvv", "scalar"):
        raise ValueError("DAG target must be rvv or scalar")
    nodes, expressions, shapes, fused = plan(graph, schedule)
    c = graph.contract
    locations = {name: f"inputs[{i}]" for i, name in enumerate(c.inputs)}
    locations.update({name: f"outputs[{i}]" for i, name in enumerate(c.outputs)})
    for i, node in enumerate(nodes):
        locations.setdefault(node.name, f"b{i}")
    code = ["/* AgentVec checked DAG; independent validation is required before deployment. */",
            "#include <stdint.h>", "#include <stddef.h>", "#include <stdlib.h>", "#include <math.h>"]
    if target == "rvv":
        code.append("#include <riscv_vector.h>")
    code += ["static int overlap(const float *a, size_t an, const float *b, size_t bn) {",
             "  uintptr_t x=(uintptr_t)a, y=(uintptr_t)b;",
             "  if (x > UINTPTR_MAX-an*4 || y > UINTPTR_MAX-bn*4) return 1;",
             "  return x < y+bn*4 && y < x+an*4;", "}",
             "int agentvec_dag(const float *const *inputs, float *const *outputs, size_t n) {",
             f"  if (!inputs || !outputs || n < {c.min_n} || n > {c.max_n} || n > SIZE_MAX/sizeof(float)) return 1;"]
    for name in c.inputs:
        code.append(f"  if (!{locations[name]}) return 1;")
    outs = list(c.outputs)
    for i, name in enumerate(outs):
        code.append(f"  if (!{locations[name]}) return 1;")
        for other in list(c.inputs) + outs[:i]:
            na = "n" if shapes[name] == "n" else "1"
            nb = "n" if shapes[other] == "n" else "1"
            code.append(f"  if (overlap({locations[name]}, {na}, {locations[other]}, {nb})) return 2;")
    temporary = [node for node in nodes if node.name not in c.outputs]
    for node in temporary:
        size = "n" if shapes[node.name] == "n" else "1"
        code.append(f"  float *{locations[node.name]} = (float*)malloc({size} * sizeof(float));")
    if temporary:
        code.append("  if (" + " || ".join(f"!{locations[node.name]}" for node in temporary) + ") {")
        code += [f"    free({locations[node.name]});" for node in temporary]
        code += ["    return 3;", "  }"]
    serial = 0

    def value(expr: DagExpr, vector: bool):
        nonlocal serial
        serial += 1
        var = f"t{serial}"
        lane = f"lane{serial}"
        args = [value(a, vector) for a in expr.args]
        if expr.op == "read":
            src = locations[expr.name]
            if vector:
                rhs = (f"__riscv_vle32_v_f32m1({src}+i, vl)" if shapes[expr.name] == "n"
                       else f"__riscv_vfmv_v_f_f32m1({src}[0], vl)")
            else:
                rhs = f"{src}[{'i' if shapes[expr.name] == 'n' else '0'}]"
        elif expr.op in ("constant", "length"):
            rhs = "(float)n" if expr.op == "length" else float(expr.value).hex() + "f"
            if vector:
                rhs = f"__riscv_vfmv_v_f_f32m1({rhs}, vl)"
        elif vector:
            binary = {"add": "vfadd", "sub": "vfsub", "mul": "vfmul", "div": "vfdiv",
                      "min": "vfmin", "max": "vfmax"}
            unary = {"abs": "vfabs", "sqrt": "vfsqrt", "neg": "vfneg"}
            if expr.op in binary:
                rhs = f"__riscv_{binary[expr.op]}_vv_f32m1({args[0]}, {args[1]}, vl)"
            elif expr.op in unary:
                rhs = f"__riscv_{unary[expr.op]}_v_f32m1({args[0]}, vl)"
            elif expr.op == "exp":
                # RVV has no exp intrinsic. Keep libm semantics for this operation.
                code.extend([f"    float {lane}[vl];",
                             f"    __riscv_vse32_v_f32m1({lane}, {args[0]}, vl);",
                             f"    for (size_t j=0; j<vl; ++j) {lane}[j]=expf({lane}[j]);"])
                rhs = f"__riscv_vle32_v_f32m1({lane}, vl)"
            else:
                raise ValueError("unsupported vector expression")
        else:
            signs = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            if expr.op in signs:
                rhs = f"({args[0]} {signs[expr.op]} {args[1]})"
            elif expr.op == "neg":
                rhs = "-" + args[0]
            else:
                fn = {"exp": "expf", "sqrt": "sqrtf", "abs": "fabsf", "min": "fminf", "max": "fmaxf"}[expr.op]
                rhs = f"{fn}({', '.join(args)})"
        code.append(f"    {'vfloat32m1_t' if vector else 'float'} {var} = {rhs};")
        return var

    for index, node in enumerate(nodes):
        code.append(f"  /* Stage {index + 1}: {node.pattern} {node.name}. */")
        vector = target == "rvv" and (shapes[node.name] == "n" or node.pattern != "map")
        reduction = node.pattern != "map"
        dst = locations[node.name]
        identity = {"sum": "0.0f", "min": "INFINITY", "max": "-INFINITY"}.get(node.reduce_op)
        code.append("  {")
        if reduction:
            if vector:
                code += ["    size_t vlmax=__riscv_vsetvlmax_e32m1();",
                         f"    vfloat32m1_t acc=__riscv_vfmv_v_f_f32m1({identity}, vlmax);"]
            else:
                code.append(f"    float acc={identity};")
        if shapes[node.name] == "n" or reduction:
            code.append("    for (size_t i=0; i<n;) {")
            if vector:
                code.append("    size_t vl=__riscv_vsetvl_e32m1(n-i);")
            result = value(expressions[node.name], vector)
            if reduction:
                if vector:
                    fn = {"sum": "vfadd", "min": "vfmin", "max": "vfmax"}[node.reduce_op]
                    code.append(f"    acc=__riscv_{fn}_vv_f32m1_tu(acc, acc, {result}, vl);")
                else:
                    rhs = f"acc+{result}" if node.reduce_op == "sum" else f"f{node.reduce_op}f(acc,{result})"
                    code.append(f"    acc={rhs};")
            else:
                code.append(f"    __riscv_vse32_v_f32m1({dst}+i, {result}, vl);" if vector else f"    {dst}[i]={result};")
            code += ["    i += vl;" if vector else "    ++i;", "    }"]
        else:
            result = value(expressions[node.name], False)
            code.append(f"    {dst}[0]={result};")
        if reduction:
            if vector:
                fn = {"sum": "vfredusum", "min": "vfredmin", "max": "vfredmax"}[node.reduce_op]
                code += [f"    vfloat32m1_t seed=__riscv_vfmv_v_f_f32m1({identity}, 1);",
                         f"    seed=__riscv_{fn}_vs_f32m1_f32m1(acc, seed, vlmax);",
                         f"    __riscv_vse32_v_f32m1({dst}, seed, 1);"]
            else:
                code.append(f"    {dst}[0]=acc;")
        code.append("  }")
    code += [f"  free({locations[node.name]});" for node in temporary]
    code += ["  return 0;", "}"]
    return "\n".join(code) + "\n", {
        "status": "CHECKED", "target": target, "graph_sha256": graph.fingerprint(),
        "topological_order": [n.name for n in nodes], "fused_edges": fused,
        "scalar_operations": ["expf"] if any("exp" in n.formula for n in nodes) else [],
        "schedule": {"fuse_maps": schedule.fuse_maps, "lmul": 1 if target == "rvv" else None},
    }
