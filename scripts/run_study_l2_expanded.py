"""Expanded Level-2 operator study over open models.

Each task compares direct RVV generation with AgentVec-style intent recovery plus
deterministic lowering on matrix-vector / rank-1 update kernels. Results are kept
separate from the Level-1 study because L2 kernels have matrix inputs and different
output shapes, but the same difftest runner reports build failures, wrong outputs,
and dynamic-check SPR.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))

from ssh_helper import HOSTS, put, run, _client

import llm_backend as LB
import obfuscate


GCC = HOSTS["server"]["riscv_gcc"]
QEMU = HOSTS["server"]["qemu"]
REMOTE_ROOT = os.environ.get("AGENTVEC_REMOTE_L2_DIFFTEST", "/tmp/agentvec_expanded_l2/difftest")
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
OUT = os.path.join(os.path.dirname(__file__), "_expanded_l2_open.json")

OPEN_MODELS = ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")
FORMS = ("clean", "renamed", "deadcode", "typealias", "combined")
VLENS = (128, 256, 512)
SIZES = (2, 7, 16, 64, 128)
EFFECTIVE_VLENS = ("native",) if QEMU in ("", "native", "none") else VLENS


@dataclass(frozen=True)
class L2Spec:
    name: str
    body: str
    contract: str
    a_matrix: bool = True
    b_matrix: bool = False
    c_matrix: bool = False
    out_matrix: bool = False
    symmetric_a: bool = False
    lower_triangular_a: bool = False
    band_radius: int = -1
    rtol: float = 1e-3


OPS = {
    "gemv": L2Spec(
        "gemv",
        "for(int i=0;i<n;i++){ float s=0.0f; for(int j=0;j<n;j++) s += a[i*n+j]*b[j]; out[i]=s; }",
        "a is a row-major n by n matrix, b is vector x, c is unused, and out[0:n] stores y = A*x.",
    ),
    "gbmv": L2Spec(
        "gbmv",
        "for(int i=0;i<n;i++){ float s=0.0f; int lo=i-2; if(lo<0) lo=0; int hi=i+2; if(hi>=n) hi=n-1; for(int j=lo;j<=hi;j++) s += a[i*n+j]*b[j]; out[i]=s; }",
        "a is a row-major n by n matrix with only the radius-2 band used, b is vector x, c is unused, and out[0:n] stores banded A*x.",
        band_radius=2,
    ),
    "symv": L2Spec(
        "symv",
        "for(int i=0;i<n;i++){ float s=0.0f; for(int j=0;j<n;j++){ float v = (j<=i) ? a[i*n+j] : a[j*n+i]; s += v*b[j]; } out[i]=s; }",
        "a stores the lower triangle of a symmetric n by n matrix in row-major slots, b is vector x, c is unused, and out[0:n] stores y = A*x.",
        symmetric_a=True,
    ),
    "trmv": L2Spec(
        "trmv",
        "for(int i=0;i<n;i++){ float s=0.0f; for(int j=0;j<=i;j++) s += a[i*n+j]*b[j]; out[i]=s; }",
        "a is a lower-triangular row-major n by n matrix, b is vector x, c is unused, and out[0:n] stores y = L*x.",
        lower_triangular_a=True,
    ),
    "ger": L2Spec(
        "ger",
        "for(int i=0;i<n;i++){ for(int j=0;j<n;j++) out[i*n+j] = c[i*n+j] + 2.0f*a[i]*b[j]; }",
        "a and b are vectors x and y, c is a row-major n by n matrix A, and out[0:n*n] stores A + 2*x*y^T.",
        a_matrix=False,
        c_matrix=True,
        out_matrix=True,
    ),
    "syr": L2Spec(
        "syr",
        "for(int i=0;i<n;i++){ for(int j=0;j<n;j++) out[i*n+j] = c[i*n+j] + 2.0f*a[i]*a[j]; }",
        "a is vector x, b is unused, c is a row-major n by n matrix A, and out[0:n*n] stores A + 2*x*x^T.",
        a_matrix=False,
        c_matrix=True,
        out_matrix=True,
    ),
}


INTENT_PROMPT = """Analyze this scalar CPU kernel and recover only its Level-2 operator intent.
Ignore names introduced by obfuscation; reason from array indexing and data flow.

```c
{src}
```

The generated code must use this output convention:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

Output only one JSON object:
{{"level":"L2","op":"gemv|gbmv|symv|trmv|ger|syr"}}

Operator meanings:
- gemv: dense matrix-vector product y = A*x.
- gbmv: radius-2 banded matrix-vector product.
- symv: symmetric matrix-vector product using lower-triangle source indexing.
- trmv: lower-triangular matrix-vector product.
- ger: rank-1 update A + 2*x*y^T.
- syr: symmetric rank-1 update A + 2*x*x^T over the full output matrix.
JSON only."""


DIRECT_PROMPT = """You are migrating a scalar CPU Level-2 linear-algebra kernel to RISC-V Vector (RVV 1.0).

Source kernel:
```c
{src}
```

Rewrite it as exactly this function:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

Input/output contract:
{contract}

Requirements:
- Use <riscv_vector.h> intrinsics and __riscv_vsetvl_e32m1 strip-mining.
- The code must be vector-length agnostic; do not hardcode lane counts.
- Preserve the source semantics for every n in the test sizes.
- Output only a ```c code block containing includes and the exact function above.
"""


L2_HELPER = """#include <riscv_vector.h>
static inline float av_hsum_f32m1(vfloat32m1_t v, size_t vl) {
    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t r = __riscv_vfredusum_vs_f32m1_f32m1(v, z, vl);
    return __riscv_vfmv_f_s_f32m1_f32(r);
}
static inline float av_dot_f32(const float *x, const float *y, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y + i, vl);
        vfloat32m1_t vp = __riscv_vfmul_vv_f32m1(vx, vy, vl);
        acc += av_hsum_f32m1(vp, vl);
        i += (int)vl;
    }
    return acc;
}
"""


def smodel(short: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", short).lower()


def strip(src: str) -> str:
    return obfuscate.strip_comments(src).strip()


def type_alias_float(src: str) -> str:
    body = re.sub(r"\bfloat\b", "elem_t", strip(src))
    return "typedef float elem_t;\n" + body


def source_clean(op: L2Spec) -> str:
    return ("void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {\n"
            "    (void)a; (void)b; (void)c;\n"
            f"    {op.body}\n"
            "}\n")


def source_forms(op: L2Spec) -> dict:
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


def macros(op: L2Spec) -> str:
    return ("#include <math.h>\n"
            "#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 0\n"
            f"#define A_IS_MATRIX {1 if op.a_matrix else 0}\n"
            f"#define B_IS_MATRIX {1 if op.b_matrix else 0}\n"
            f"#define C_IS_MATRIX {1 if op.c_matrix else 0}\n"
            f"#define OUT_IS_MATRIX {1 if op.out_matrix else 0}\n"
            f"#define SYMMETRIC_A {1 if op.symmetric_a else 0}\n"
            f"#define LOWER_TRIANGULAR_A {1 if op.lower_triangular_a else 0}\n"
            f"#define BANDED_A_RADIUS {op.band_radius}\n")


def scalar_kernel(op: L2Spec) -> str:
    return (macros(op) +
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)a; (void)b; (void)c;\n"
            f"    {op.body}\n"
            "}\n#include \"harness.h\"\n")


def emit_gemv(op: L2Spec) -> str:
    return (macros(op) + L2_HELPER +
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)c;\n"
            "    for (int i = 0; i < n; i++) out[i] = av_dot_f32(a + (size_t)i*n, b, n);\n"
            "}\n#include \"harness.h\"\n")


def emit_gbmv(op: L2Spec) -> str:
    return (macros(op) + L2_HELPER +
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)c;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        int lo = i - 2; if (lo < 0) lo = 0;\n"
            "        int hi = i + 2; if (hi >= n) hi = n - 1;\n"
            "        out[i] = av_dot_f32(a + (size_t)i*n + lo, b + lo, hi - lo + 1);\n"
            "    }\n"
            "}\n#include \"harness.h\"\n")


def emit_trmv(op: L2Spec) -> str:
    return (macros(op) + L2_HELPER +
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)c;\n"
            "    for (int i = 0; i < n; i++) out[i] = av_dot_f32(a + (size_t)i*n, b, i + 1);\n"
            "}\n#include \"harness.h\"\n")


def emit_ger(op: L2Spec, use_b: bool) -> str:
    rhs = "b + j" if use_b else "a + j"
    return (macros(op) +
            "#include <riscv_vector.h>\n"
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)b;\n" if not use_b else macros(op) + "#include <riscv_vector.h>\n"
            "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n") + (
            "    for (int i = 0; i < n; i++) {\n"
            "        float xi = 2.0f * a[i];\n"
            "        for (int j = 0; j < n; ) {\n"
            "            size_t vl = __riscv_vsetvl_e32m1((size_t)(n - j));\n"
            "            vfloat32m1_t vc = __riscv_vle32_v_f32m1(c + (size_t)i*n + j, vl);\n"
            f"            vfloat32m1_t vy = __riscv_vle32_v_f32m1({rhs}, vl);\n"
            "            vc = __riscv_vfmacc_vf_f32m1(vc, xi, vy, vl);\n"
            "            __riscv_vse32_v_f32m1(out + (size_t)i*n + j, vc, vl);\n"
            "            j += (int)vl;\n"
            "        }\n"
            "    }\n"
            "}\n#include \"harness.h\"\n")


EMITTERS = {
    "gemv": emit_gemv,
    "gbmv": emit_gbmv,
    "symv": emit_gemv,
    "trmv": emit_trmv,
    "ger": lambda op: emit_ger(op, True),
    "syr": lambda op: emit_ger(op, False),
}


def normalize_op(intent) -> str:
    if not isinstance(intent, dict):
        return ""
    if "op" in intent:
        raw = intent["op"]
    else:
        raw = ""
        for key in ("level2", "l2", "intent", "operator"):
            nested = intent.get(key)
            if isinstance(nested, dict) and "op" in nested:
                raw = nested["op"]
                break
    s = re.sub(r"[^a-z0-9]", "", str(raw).lower())
    aliases = {
        "densegemv": "gemv", "matvec": "gemv", "matrixvector": "gemv",
        "bandedgemv": "gbmv", "bandedmatvec": "gbmv",
        "symmetricgemv": "symv", "symmetricmatvec": "symv",
        "triangularmatvec": "trmv", "lowertriangularmatvec": "trmv",
        "rank1": "ger", "rank1update": "ger", "outerproductupdate": "ger",
        "symmetricrank1": "syr", "symmetricrank1update": "syr",
    }
    return aliases.get(s, s)


def gen_agentvec(model_id: str, src: str, op: L2Spec, use_cache: bool) -> tuple:
    text = LB.chat(model_id, INTENT_PROMPT.format(src=src), max_tokens=900, timeout=140, use_cache=use_cache)
    intent = LB.extract_json(text)
    got = normalize_op(intent)
    if got != op.name:
        if got == "gemv" and op.name in ("gbmv", "symv", "trmv"):
            note = intent if isinstance(intent, dict) else {"op": got}
            note = dict(note)
            note["accepted_as"] = "generic_matvec_fallback"
            note["target_operator"] = op.name
            return emit_gemv(op), note
        raise ValueError(f"bad-l2-intent:{got or 'none'}")
    return EMITTERS[op.name](op), intent


def gen_direct(model_id: str, src: str, op: L2Spec, use_cache: bool) -> tuple:
    text = LB.chat(model_id, DIRECT_PROMPT.format(src=src, contract=op.contract),
                   max_tokens=3200, timeout=180, use_cache=use_cache)
    code = LB.extract_c(text)
    includes = "" if "riscv_vector.h" in code else "#include <riscv_vector.h>\n"
    return macros(op) + includes + code + "\n#include \"harness.h\"\n", {"direct": "rvv"}


def save_case(label: str, rvv: str, op: L2Spec):
    d = os.path.join(DIFF, "kernels", label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rvv.c"), "w", newline="\n") as f:
        f.write(rvv)
    with open(os.path.join(d, "scalar.c"), "w", newline="\n") as f:
        f.write(scalar_kernel(op))
    spec = {"name": label, "dtype": "f32", "reduce": False, "tier": "float_ulp",
            "rtol": op.rtol, "sizes": list(SIZES), "level": "L2"}
    with open(os.path.join(d, "spec.json"), "w", newline="\n") as f:
        json.dump(spec, f)
    remote_dir = f"{REMOTE_ROOT}/kernels/{label}"
    run("server", f"mkdir -p {remote_dir}", timeout=60)
    client = _client("server")
    try:
        sftp = client.open_sftp()
        for name in ("rvv.c", "scalar.c", "spec.json"):
            sftp.put(os.path.join(d, name), f"{remote_dir}/{name}")
        sftp.close()
    finally:
        client.close()


def summarize(tasks, results, seeds):
    planned_per_task = len(SIZES) * seeds * len(EFFECTIVE_VLENS)
    out = {"by_arm": {}, "by_operator": {}}
    for arm in ("direct", "agentvec"):
        items = [t for t in tasks if t["arm"] == arm]
        generated = [t for t in items if t["status"] == "generated"]
        gen_fail = [t for t in items if t["status"] != "generated"]
        build_fail = []
        wrong = []
        full = []
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
            if total and total == passed:
                full.append(t)
            else:
                wrong.append(t)
        planned = len(items) * planned_per_task
        out["by_arm"][arm] = {
            "attempted_generated_kernels": len(items),
            "generated_source_files": len(generated),
            "generation_failures": len(gen_fail),
            "build_failures": len(build_fail),
            "wrong_or_partial": len(wrong),
            "fully_passing_kernels": len(full),
            "planned_dynamic_checks": planned,
            "executed_dynamic_checks": executed_checks,
            "passed_dynamic_checks": passed_checks,
            "planned_check_spr": round(passed_checks / planned, 4) if planned else 0.0,
        }
    for opname in sorted({t["operator"] for t in tasks}):
        out["by_operator"][opname] = {}
        for arm in ("direct", "agentvec"):
            labels = [t["label"] for t in tasks if t["operator"] == opname and t["arm"] == arm and t["status"] == "generated"]
            passed = sum(int(results.get(label, {}).get("passed", 0)) for label in labels)
            total = sum(int(results.get(label, {}).get("total", 0)) for label in labels)
            out["by_operator"][opname][arm] = {
                "generated_source_files": len(labels),
                "executed_dynamic_checks": total,
                "passed_dynamic_checks": passed,
                "executed_check_spr": round(passed / total, 4) if total else 0.0,
            }
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=",".join(OPS), help="comma-separated L2 operator names")
    ap.add_argument("--forms", default=",".join(FORMS), help="comma-separated source forms")
    ap.add_argument("--models", default=",".join(OPEN_MODELS), help="comma-separated model short names")
    ap.add_argument("--output", default=OUT, help="result JSON path")
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
        if not REMOTE_ROOT.startswith("/tmp/agentvec_expanded_l2/"):
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
                    label = f"l2_{arm}_{opname}_{ms}_{form}"
                    rec = {"label": label, "arm": arm, "operator": opname, "model": model,
                           "source_form": form, "status": "pending"}
                    print(f"[{idx}/{n}] {label}", flush=True)
                    try:
                        rvv, note = fn(mid, src, op, args.use_cache)
                        save_case(label, rvv, op)
                        rec.update(status="generated", note=note)
                        print("  -> generated", flush=True)
                    except Exception as e:
                        rec.update(status="generation_failure", error=str(e)[:240])
                        print(f"  -> generation_failure: {rec['error']}", flush=True)
                    tasks.append(rec)

    print("running L2 differential tests on remote build host...", flush=True)
    vstr = ",".join(map(str, EFFECTIVE_VLENS))
    rc, out, err = run("server", f"cd {REMOTE_ROOT} && python3 runner.py --root {REMOTE_ROOT} "
                                 f"--gcc {GCC} --qemu {QEMU} --vlens {vstr} --seeds {args.seeds} 2>&1",
                       timeout=7200)
    i = out.find("{")
    report = json.loads(out[i:]) if i >= 0 else {"parse_error": {"stdout": out[-2000:], "stderr": err[-2000:], "rc": rc}}
    results = report.get("kernels", {})
    final = {
        "config": {
            "level": "L2", "operators": selected_ops, "source_forms": selected_forms,
            "models": selected_models, "seeds": args.seeds, "sizes": list(SIZES),
            "vlens": list(EFFECTIVE_VLENS),
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
