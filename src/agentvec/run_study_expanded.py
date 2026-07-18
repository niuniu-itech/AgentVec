"""Expanded direct-generation study over Level-1/idiom operators.

This replaces the earlier three-reduction slice with a broader, auditable suite.
For each operator, each open model sees multiple source forms and must either:
  (1) directly emit RVV VLA C, or
  (2) recover AIR intent that deterministic AgentVec lowering turns into RVV.

The result file reports attempted generated kernels and planned dynamic checks, so
build/generation failures remain visible in the denominator.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))

from ssh_helper import HOSTS, put, run

import llm_backend as LB
import obfuscate
from air_from_formula import air_from_intent
from guard import guard
from lowering import emit_reduce_kernel, emit_rvv_kernel


GCC = HOSTS["server"]["riscv_gcc"]
QEMU = HOSTS["server"]["qemu"]
REMOTE_ROOT = os.environ.get("AGENTVEC_REMOTE_DIFFTEST", "/tmp/agentvec_expanded/difftest")
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
OUT = os.path.join(os.path.dirname(__file__), "_expanded_open.json")

OPEN_MODELS = ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")
FORMS = ("clean", "renamed", "deadcode", "typealias", "combined")
VLENS = (128, 256, 512)
SIZES = (1, 7, 64, 1000, 4096)
EFFECTIVE_VLENS = ("native",) if QEMU in ("", "native", "none") else VLENS


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    body: str
    rtol: float = 1e-6
    out_factor: int = 1
    reduce_op: str = ""
    pre: str = ""
    post: str = None
    formula: str = ""


OPS = {
    "axpy":  OpSpec("axpy",  "map",  "for(int i=0;i<n;i++) out[i] = 2.0f*a[i] + b[i];", formula="out = 2*a + b"),
    "axpby": OpSpec("axpby", "map",  "for(int i=0;i<n;i++) out[i] = 2.0f*a[i] + 3.0f*b[i];", formula="out = 2*a + 3*b"),
    "copy":  OpSpec("copy",  "map",  "for(int i=0;i<n;i++) out[i] = a[i];", formula="out = a"),
    "scal":  OpSpec("scal",  "map",  "for(int i=0;i<n;i++) out[i] = 2.0f*a[i];", formula="out = 2*a"),
    "swap":  OpSpec("swap",  "pair", "for(int i=0;i<n;i++){ out[i] = b[i]; out[n+i] = a[i]; }", out_factor=2),
    "dot":   OpSpec("dot",   "reduce", "float s=0.0f; for(int i=0;i<n;i++) s += a[i]*b[i]; out[0]=s;",
                     rtol=1e-4, reduce_op="sum", pre="ab"),
    "asum":  OpSpec("asum",  "reduce", "float s=0.0f; for(int i=0;i<n;i++) s += fabsf(a[i]); out[0]=s;",
                     rtol=1e-4, reduce_op="sum", pre="abs_a"),
    "nrm2":  OpSpec("nrm2",  "reduce", "float s=0.0f; for(int i=0;i<n;i++) s += a[i]*a[i]; out[0]=sqrtf(s);",
                     rtol=1e-4, reduce_op="sum", pre="aa", post="sqrt"),
    "max":   OpSpec("max",   "reduce", "float s=a[0]; for(int i=1;i<n;i++) if(a[i]>s) s=a[i]; out[0]=s;",
                     rtol=1e-6, reduce_op="max", pre="a"),
    "min":   OpSpec("min",   "reduce", "float s=a[0]; for(int i=1;i<n;i++) if(a[i]<s) s=a[i]; out[0]=s;",
                     rtol=1e-6, reduce_op="min", pre="a"),
}


INTENT_PROMPT = """Analyze this scalar CPU kernel and recover only its algorithmic intent.
Ignore names introduced by obfuscation; reason from data flow.

```c
{src}
```

The target convention names the source arrays a, b, c in parameter order.
Output only one JSON object:
- elementwise map: {{"pattern":"map","dtype":"f32","formula":"out = <expr over a,b,c, constants, + - *>"}}
- paired map:      {{"pattern":"pairmap","dtype":"f32","out0":"b","out1":"a"}}
                  meaning out[0:n]=b and out[n:2n]=a
- reduction:       {{"pattern":"reduce","dtype":"f32","reduce_op":"sum|max|min",
                    "elem":"a|a*b|abs(a)|a*a","postproc":"none|sqrt"}}
JSON only."""


DIRECT_PROMPT = """You are migrating a scalar CPU kernel to RISC-V Vector (RVV 1.0).

Source kernel:
```c
{src}
```

Rewrite it as exactly this function:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

The inputs a, b, c correspond to the source arrays in parameter order. The output contract is:
{out_contract}

Requirements:
- Use <riscv_vector.h> intrinsics and __riscv_vsetvl_e32m1 strip-mining.
- The code must be vector-length agnostic; do not hardcode lane counts.
- Preserve the source semantics for every n in the test sizes.
- Output only a ```c code block containing includes and the exact function above.
"""


def smodel(short: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", short).lower()


def strip(src: str) -> str:
    return obfuscate.strip_comments(src).strip()


def type_alias_float(src: str) -> str:
    body = re.sub(r"\bfloat\b", "elem_t", strip(src))
    return "typedef float elem_t;\n" + body


def source_clean(op: OpSpec) -> str:
    return ("#include <math.h>\n"
            "void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {\n"
            "    (void)c;\n"
            "    (void)b;\n"
            f"    {op.body}\n"
            "}\n")


def source_forms(op: OpSpec) -> dict:
    clean = source_clean(op)
    typed = type_alias_float(clean)
    combined = obfuscate.identifier_mangling(obfuscate.dead_code_insertion(typed))
    return {
        "clean": strip(clean),
        "renamed": obfuscate.identifier_mangling(strip(clean)),
        "deadcode": obfuscate.dead_code_insertion(strip(clean)),
        "typealias": typed,
        "combined": combined,
    }


def macros(op: OpSpec) -> str:
    return ("#include <math.h>\n"
            "#define DT float\n#define DT_IS_FLOAT 1\n"
            f"#define REDUCE {1 if op.kind == 'reduce' else 0}\n"
            f"#define OUT_FACTOR {op.out_factor}\n")


def scalar_kernel(op: OpSpec) -> str:
    return (macros(op) +
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)b; (void)c;\n"
            f"    {op.body}\n"
            "}\n#include \"harness.h\"\n")


def emit_swap_kernel() -> str:
    return (macros(OPS["swap"]) +
            "#include <riscv_vector.h>\n"
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)c;\n"
            "    size_t vl;\n"
            "    for (int i = 0; i < n; i += (int)vl) {\n"
            "        vl = __riscv_vsetvl_e32m1((size_t)(n - i));\n"
            "        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);\n"
            "        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);\n"
            "        __riscv_vse32_v_f32m1(out + i, vb, vl);\n"
            "        __riscv_vse32_v_f32m1(out + n + i, va, vl);\n"
            "    }\n"
            "}\n#include \"harness.h\"\n")


def intent_to_reduce(intent: dict):
    rop = str(intent.get("reduce_op", "sum")).lower()
    elem = str(intent.get("elem") or intent.get("formula") or "a").replace(" ", "").lower().split("=")[-1]
    pre = {
        "a": "a", "a[i]": "a",
        "a*b": "ab", "b*a": "ab", "a[i]*b[i]": "ab",
        "abs(a)": "abs_a", "fabs(a)": "abs_a", "fabsf(a)": "abs_a", "|a|": "abs_a",
        "a*a": "aa", "a^2": "aa", "a**2": "aa", "a[i]*a[i]": "aa",
    }.get(elem)
    post = "sqrt" if "sqrt" in str(intent.get("postproc", "")).lower() else None
    return rop, pre, post


def gen_agentvec(model_id: str, src: str, op: OpSpec, use_cache: bool) -> tuple:
    text = LB.chat(model_id, INTENT_PROMPT.format(src=src), max_tokens=1200, timeout=140, use_cache=use_cache)
    intent = LB.extract_json(text)
    if not intent:
        raise ValueError("no-json")
    if "pattern" not in intent:
        for key in ("elementwise_map", "map", "paired_map", "pairmap", "reduction", "reduce"):
            nested = intent.get(key)
            if isinstance(nested, dict):
                intent = nested
                break
    pat = intent.get("pattern")
    if not pat:
        if "formula" in intent:
            intent["pattern"] = pat = "map"
        elif "out0" in intent and "out1" in intent:
            intent["pattern"] = pat = "pairmap"
        elif "elem" in intent or "reduce_op" in intent:
            intent["pattern"] = pat = "reduce"
    if pat == "map":
        air = air_from_intent(intent)
        ok, proof = guard(air)
        if not ok:
            raise ValueError(f"veto:{proof[:50]}")
        return emit_rvv_kernel(air), intent
    if pat == "pairmap":
        out0 = str(intent.get("out0", "")).strip().lower()
        out1 = str(intent.get("out1", "")).strip().lower()
        if out0 != "b" or out1 != "a":
            raise ValueError(f"bad-pairmap:{out0}/{out1}")
        return emit_swap_kernel(), intent
    if pat == "reduce":
        rop, pre, post = intent_to_reduce(intent)
        if pre is None or rop not in ("sum", "max", "min"):
            raise ValueError(f"bad-reduce:{rop}/{intent.get('elem')}")
        return emit_reduce_kernel(rop, pre, post), intent
    raise ValueError(f"bad-pattern:{pat}")


def gen_direct(model_id: str, src: str, op: OpSpec, use_cache: bool) -> tuple:
    if op.kind == "reduce":
        contract = "write one scalar result to out[0]."
    elif op.kind == "pair":
        contract = "write 2*n elements: out[0:n] is the first swapped stream and out[n:2*n] is the second."
    else:
        contract = "write n elements to out[0:n]."
    text = LB.chat(model_id, DIRECT_PROMPT.format(src=src, out_contract=contract),
                   max_tokens=2600, timeout=160, use_cache=use_cache)
    code = LB.extract_c(text)
    includes = "" if "riscv_vector.h" in code else "#include <riscv_vector.h>\n"
    return macros(op) + includes + code + "\n#include \"harness.h\"\n", {"direct": "rvv"}


def save_case(label: str, rvv: str, op: OpSpec):
    d = os.path.join(DIFF, "kernels", label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rvv.c"), "w", newline="\n") as f:
        f.write(rvv)
    with open(os.path.join(d, "scalar.c"), "w", newline="\n") as f:
        f.write(scalar_kernel(op))
    spec = {"name": label, "dtype": "f32", "reduce": op.kind == "reduce", "tier": "float_ulp",
            "rtol": op.rtol, "sizes": list(SIZES)}
    with open(os.path.join(d, "spec.json"), "w", newline="\n") as f:
        json.dump(spec, f)
    for name in ("rvv.c", "scalar.c", "spec.json"):
        put("server", os.path.join(d, name), f"{REMOTE_ROOT}/kernels/{label}/{name}")


def summarize(tasks, results, seeds):
    planned_per_task = len(SIZES) * seeds * len(EFFECTIVE_VLENS)
    out = {}
    for arm in ("direct", "agentvec"):
        items = [t for t in tasks if t["arm"] == arm]
        generated = [t for t in items if t["status"] == "generated"]
        gen_fail = [t for t in items if t["status"] != "generated"]
        build_fail = []
        wrong = []
        pass_tasks = []
        passed_checks = 0
        executed_checks = 0
        for t in generated:
            r = results.get(t["label"], {})
            if r.get("build_error"):
                build_fail.append(t)
                continue
            total = int(r.get("total", 0))
            passed = int(r.get("passed", 0))
            executed_checks += total
            passed_checks += passed
            if total and passed == total:
                pass_tasks.append(t)
            else:
                wrong.append(t)
        planned = len(items) * planned_per_task
        out[arm] = {
            "attempted_generated_kernels": len(items),
            "generated_source_files": len(generated),
            "generation_failures": len(gen_fail),
            "build_failures": len(build_fail),
            "wrong_or_partial": len(wrong),
            "fully_passing_kernels": len(pass_tasks),
            "planned_dynamic_checks": planned,
            "executed_dynamic_checks": executed_checks,
            "passed_dynamic_checks": passed_checks,
            "planned_check_spr": round(passed_checks / planned, 4) if planned else 0.0,
        }
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=",".join(OPS), help="comma-separated operator names")
    ap.add_argument("--forms", default=",".join(FORMS), help="comma-separated source forms")
    ap.add_argument("--models", default=",".join(OPEN_MODELS), help="comma-separated model short names")
    ap.add_argument("--output", default=OUT, help="result JSON path; defaults to a new-model result file")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--use-cache", action="store_true", help="reuse cached LLM responses")
    ap.add_argument("--no-clean-remote", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    selected_ops = [o for o in args.ops.split(",") if o]
    selected_forms = [f for f in args.forms.split(",") if f]
    selected_models = [m for m in args.models.split(",") if m]

    if not args.no_clean_remote:
        if not REMOTE_ROOT.startswith("/tmp/agentvec_expanded/"):
            raise RuntimeError(f"refusing to clean non-temp remote root: {REMOTE_ROOT}")
        run("server", f"rm -rf {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}/kernels", timeout=60)
    put("server", os.path.join(DIFF, "runner.py"), f"{REMOTE_ROOT}/runner.py")
    put("server", os.path.join(DIFF, "harness.h"), f"{REMOTE_ROOT}/harness.h")

    tasks = []
    n = len(selected_ops) * len(selected_forms) * len(selected_models) * 2
    idx = 0
    for opname in selected_ops:
        op = OPS[opname]
        forms = source_forms(op)
        for form in selected_forms:
            src = forms[form]
            for model in selected_models:
                mid = LB.MODELS[model]
                ms = smodel(model)
                for arm, fn in (("direct", gen_direct), ("agentvec", gen_agentvec)):
                    idx += 1
                    label = f"ex_{arm}_{opname}_{ms}_{form}"
                    rec = {"label": label, "arm": arm, "operator": opname, "model": model,
                           "source_form": form, "status": "pending"}
                    print(f"[{idx}/{n}] {label}", flush=True)
                    try:
                        rvv, note = fn(mid, src, op, args.use_cache)
                        save_case(label, rvv, op)
                        rec.update(status="generated", note=note)
                        print(f"  -> generated", flush=True)
                    except Exception as e:
                        rec.update(status="generation_failure", error=str(e)[:240])
                        print(f"  -> generation_failure: {rec['error']}", flush=True)
                    tasks.append(rec)

    print("running cross-VLEN differential tests on remote build host...", flush=True)
    vstr = ",".join(map(str, EFFECTIVE_VLENS))
    rc, out, err = run("server", f"cd {REMOTE_ROOT} && python3 runner.py --root {REMOTE_ROOT} "
                                 f"--gcc {GCC} --qemu {QEMU} --vlens {vstr} --seeds {args.seeds} 2>&1",
                       timeout=7200)
    i = out.find("{")
    report = json.loads(out[i:]) if i >= 0 else {"parse_error": {"stdout": out[-2000:], "stderr": err[-2000:], "rc": rc}}
    results = report.get("kernels", {})
    final = {
        "config": {
            "operators": selected_ops, "source_forms": selected_forms, "models": selected_models,
            "seeds": args.seeds, "sizes": list(SIZES), "vlens": list(EFFECTIVE_VLENS),
            "dynamic_checks_per_kernel": len(SIZES) * args.seeds * len(EFFECTIVE_VLENS),
            "remote_root": REMOTE_ROOT,
        },
        "tasks": tasks,
        "runner_report": report,
        "summary": summarize(tasks, results, args.seeds),
    }
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    with open(out_path, "w", newline="\n", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final["summary"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
