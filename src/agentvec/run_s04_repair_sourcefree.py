"""S04 probes for noisy-code recovery and source-free PTX/SASS evidence.

This script fills the two planned rows in the S03 supplement:

  1. Noisy-code recovery.  A correct operator implementation is first lowered to
     RVV, then corrupted with API-name, API-width, include, or syntax faults.
     AgentVec receives the faulty code as evidence and recovers AIR intent; the
     direct baseline tries to repair target code directly.

  2. PTX/SASS-only source-free recovery.  The source is hidden and the model sees
     only PTX-like or SASS-like evidence.  AgentVec again returns intent, whereas
     the direct baseline emits RVV code.

Both probes cover the same 16 operator families as the expanded L1/L2 source
stress tests: 10 L1/idiom operators and 6 L2 operators.  Results are reported in
the same denominator style as the existing expanded studies: generation failures,
build failures, wrong/partial kernels, and planned-check SPR.
"""
import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parent
DIFF = PLATFORM / "03-rvv-difftest-suite"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PLATFORM / "04-riscv-lab-support"))

import llm_backend as LB
from air_from_formula import air_from_intent
from guard import guard
from lowering import emit_reduce_kernel, emit_rvv_kernel
from ssh_helper import HOSTS, put, run

L1 = importlib.import_module("run_study_expanded")
L2 = importlib.import_module("run_study_l2_expanded")

REMOTE_ROOT = os.environ.get("AGENTVEC_S04_REMOTE_ROOT", "/tmp/agentvec_s04_extra/difftest")
GCC = os.environ.get("AGENTVEC_SERVER_GCC", HOSTS["server"]["riscv_gcc"])
QEMU = os.environ.get("AGENTVEC_SERVER_QEMU", HOSTS["server"]["qemu"])
EFFECTIVE_VLENS = ("native",) if QEMU in ("", "native", "none") else (128, 256, 512)

DEFAULT_MODELS = ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")
L1_OPS = tuple(L1.OPS.keys())
L2_OPS = tuple(L2.OPS.keys())
FAULTS = ("api_name", "api_width", "missing_include", "syntax")
VIEWS = ("ptx", "sass")


def smodel(short: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", short).lower()


def macros(op, level: str) -> str:
    return L1.macros(op) if level == "L1" else L2.macros(op)


def scalar_kernel(op, level: str) -> str:
    return L1.scalar_kernel(op) if level == "L1" else L2.scalar_kernel(op)


def spec_for(op, level: str, label: str) -> dict:
    sizes = list(L1.SIZES if level == "L1" else L2.SIZES)
    return {
        "name": label,
        "dtype": "f32",
        "reduce": bool(level == "L1" and op.kind == "reduce"),
        "tier": "float_ulp",
        "rtol": float(getattr(op, "rtol", 1e-4)),
        "sizes": sizes,
        "level": level,
    }


def l1_valid_rvv(op) -> str:
    if op.kind == "pair":
        return L1.emit_swap_kernel()
    if op.kind == "reduce":
        return emit_reduce_kernel(op.reduce_op, op.pre, op.post)
    intent = {"pattern": "map", "dtype": "f32", "formula": op.formula}
    air = air_from_intent(intent)
    ok, proof = guard(air)
    if not ok:
        raise ValueError(proof)
    return emit_rvv_kernel(air)


def valid_rvv(op, level: str) -> str:
    if level == "L1":
        return l1_valid_rvv(op)
    return L2.EMITTERS[op.name](op)


def inject_fault(code: str, fault: str) -> str:
    if fault == "api_name":
        return code.replace("__riscv_vsetvl_e32m1", "__riscv_vsetvl_e32m1_BAD", 1)
    if fault == "api_width":
        out = code.replace("__riscv_vle32_v_f32m1", "__riscv_vle64_v_f32m1", 1)
        return out if out != code else code.replace("__riscv_vse32_v_f32m1", "__riscv_vse64_v_f32m1", 1)
    if fault == "missing_include":
        return code.replace("#include <riscv_vector.h>\n", "", 1)
    if fault == "syntax":
        return code.replace(";", "", 1)
    raise ValueError(f"unknown fault {fault}")


L1_AIR_FROM_BROKEN = """Recover the algorithmic intent from this faulty RVV-like operator code.
The code may contain API-name, vector-width, include, or syntax faults.  Do not
repair syntax.  Infer the intended computation from data flow and return only one
JSON object:
- map:      {{"pattern":"map","dtype":"f32","formula":"out = <expr over a,b,c>"}}
- pairmap:  {{"pattern":"pairmap","dtype":"f32","out0":"b","out1":"a"}}
- reduce:   {{"pattern":"reduce","dtype":"f32","reduce_op":"sum|max|min",
              "elem":"a|a*b|abs(a)|a*a","postproc":"none|sqrt"}}

Faulty evidence:
```c
{evidence}
```
JSON only."""


L2_AIR_FROM_BROKEN = """Recover the Level-2 operator intent from this faulty RVV-like code.
The code may contain API-name, vector-width, include, or syntax faults.  Do not
repair syntax.  Infer the intended operator from indexing and data flow.

Return only one JSON object:
{{"level":"L2","op":"gemv|gbmv|symv|trmv|ger|syr"}}

Faulty evidence:
```c
{evidence}
```
JSON only."""


DIRECT_REPAIR_L1 = """This RVV operator code is faulty. Repair it into exactly this function:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

The original operator is one of the BLAS-like L1/idiom kernels. Preserve the
intended computation, use RVV 1.0 intrinsics with vector-length-agnostic
__riscv_vsetvl_e32m1 strip-mining, and output only a ```c code block.

Faulty code:
```c
{evidence}
```"""


DIRECT_REPAIR_L2 = """This RVV Level-2 operator code is faulty. Repair it into exactly this function:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

Preserve the intended matrix/vector computation, use RVV 1.0 intrinsics with
vector-length-agnostic __riscv_vsetvl_e32m1 strip-mining, and output only a
```c code block.

Faulty code:
```c
{evidence}
```"""


L1_AIR_FROM_SOURCEFREE = """Recover the operator intent from source-free GPU evidence.
Only PTX/SASS-like low-level evidence is available; CUDA/C source is hidden.
Return only one JSON object:
- map:      {{"pattern":"map","dtype":"f32","formula":"out = <expr over a,b,c>"}}
- pairmap:  {{"pattern":"pairmap","dtype":"f32","out0":"b","out1":"a"}}
- reduce:   {{"pattern":"reduce","dtype":"f32","reduce_op":"sum|max|min",
              "elem":"a|a*b|abs(a)|a*a","postproc":"none|sqrt"}}

Evidence:
```asm
{evidence}
```
JSON only."""


L2_AIR_FROM_SOURCEFREE = """Recover the Level-2 operator intent from source-free GPU evidence.
Only PTX/SASS-like low-level evidence is available; CUDA/C source is hidden.
Return only one JSON object:
{{"level":"L2","op":"gemv|gbmv|symv|trmv|ger|syr"}}

Evidence:
```asm
{evidence}
```
JSON only."""


DIRECT_SOURCEFREE_L1 = """Migrate this source-free GPU evidence to RVV 1.0.
Emit exactly:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

Use vector-length-agnostic __riscv_vsetvl_e32m1 strip-mining. Output only a
```c code block with includes and the function.

Evidence:
```asm
{evidence}
```"""


DIRECT_SOURCEFREE_L2 = """Migrate this source-free GPU evidence to RVV 1.0.
Emit exactly:
  void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n)

The evidence describes a Level-2 matrix/vector operator. Use vector-length-
agnostic RVV strip-mining. Output only a ```c code block.

Evidence:
```asm
{evidence}
```"""


def source_free_evidence(op, level: str, view: str) -> str:
    if level == "L1":
        name = op.name
        pfx = ".visible .entry k(.param a,.param b,.param c,.param out,.param n)" if view == "ptx" else "/* SASS native assembly sketch */"
        templates = {
            "axpy":  "ld a[i]; ld b[i]; fma.rn.f32 t, a[i], 2.0, b[i]; st out[i], t",
            "axpby": "ld a[i]; ld b[i]; mul.f32 x,a[i],2.0; fma.rn.f32 t,b[i],3.0,x; st out[i],t",
            "copy":  "ld a[i]; st out[i], a[i]",
            "scal":  "ld a[i]; mul.f32 t,a[i],2.0; st out[i],t",
            "swap":  "ld a[i]; ld b[i]; st out[i], b[i]; st out[n+i], a[i]",
            "dot":   "ld a[i]; ld b[i]; mul.f32 t,a[i],b[i]; red.add.f32 acc,t; st out[0],acc",
            "asum":  "ld a[i]; abs.f32 t,a[i]; red.add.f32 acc,t; st out[0],acc",
            "nrm2":  "ld a[i]; mul.f32 t,a[i],a[i]; red.add.f32 acc,t; sqrt.rn.f32 r,acc; st out[0],r",
            "max":   "ld a[i]; red.max.f32 acc,a[i]; st out[0],acc",
            "min":   "ld a[i]; red.min.f32 acc,a[i]; st out[0],acc",
        }[name]
        if view == "sass":
            templates = (templates.replace("ld", "LDG.E")
                         .replace("st", "STG.E")
                         .replace("fma.rn.f32", "FFMA")
                         .replace("mul.f32", "FMUL")
                         .replace("red.add.f32", "FADD.RED"))
        return f"{pfx}\n// thread index i ranges over n elements\n{templates}\n// no C/CUDA source is provided"

    name = op.name
    pfx = ".visible .entry l2(.param a,.param b,.param c,.param out,.param n)" if view == "ptx" else "/* SASS native assembly sketch */"
    templates = {
        "gemv": "for row i: acc=0; for col j: ld A[i*n+j]; ld x[j]; fma acc,A,x; st out[i],acc",
        "gbmv": "for row i: acc=0; for j in [i-2,i+2] clipped: ld A[i*n+j]; ld x[j]; fma acc,A,x; st out[i],acc",
        "symv": "for row i: acc=0; for col j: select lower-triangle A[max(i,j),min(i,j)]; fma acc,A,x[j]; st out[i],acc",
        "trmv": "for row i: acc=0; for col j<=i: ld L[i*n+j]; ld x[j]; fma acc,L,x; st out[i],acc",
        "ger":  "for row i,col j: ld C[i*n+j]; ld x[i]; ld y[j]; fma t,2*x[i],y[j],C; st out[i*n+j],t",
        "syr":  "for row i,col j: ld C[i*n+j]; ld x[i]; ld x[j]; fma t,2*x[i],x[j],C; st out[i*n+j],t",
    }[name]
    if view == "sass":
        templates = templates.replace("ld", "LDG.E").replace("st", "STG.E").replace("fma", "FFMA")
    return f"{pfx}\n{templates}\n// source is hidden; infer operator from address expressions"


def l1_intent_to_rvv(intent: dict) -> str:
    if isinstance(intent.get("formula"), str):
        intent = dict(intent)
        intent["formula"] = re.sub(r"(\d+(?:\.\d+)?)f\b", r"\1", intent["formula"])
    pat = intent.get("pattern")
    if not pat:
        if "formula" in intent:
            pat = intent["pattern"] = "map"
        elif "out0" in intent:
            pat = intent["pattern"] = "pairmap"
        elif "elem" in intent or "reduce_op" in intent:
            pat = intent["pattern"] = "reduce"
    if pat == "map":
        air = air_from_intent(intent)
        ok, proof = guard(air)
        if not ok:
            raise ValueError("veto:" + proof)
        return emit_rvv_kernel(air)
    if pat == "pairmap":
        if str(intent.get("out0", "")).lower() != "b" or str(intent.get("out1", "")).lower() != "a":
            raise ValueError("bad-pairmap")
        return L1.emit_swap_kernel()
    if pat == "reduce":
        rop, pre, post = L1.intent_to_reduce(intent)
        if pre is None:
            raise ValueError("bad-reduce")
        return emit_reduce_kernel(rop, pre, post)
    raise ValueError(f"bad-pattern:{pat}")


def l2_intent_to_rvv(intent: dict, op) -> str:
    got = L2.normalize_op(intent)
    if got != op.name:
        if got == "gemv" and op.name in ("gbmv", "symv", "trmv"):
            return L2.emit_gemv(op)
        raise ValueError(f"bad-l2-intent:{got or 'none'}")
    return L2.EMITTERS[op.name](op)


def agentvec_from_evidence(model_id: str, evidence: str, op, level: str, mode: str, use_cache: bool) -> tuple[str, dict]:
    if mode == "repair":
        prompt = (L1_AIR_FROM_BROKEN if level == "L1" else L2_AIR_FROM_BROKEN).format(evidence=evidence)
    else:
        prompt = (L1_AIR_FROM_SOURCEFREE if level == "L1" else L2_AIR_FROM_SOURCEFREE).format(evidence=evidence)
    text = LB.chat(model_id, prompt, max_tokens=1200, timeout=180, use_cache=use_cache)
    intent = LB.extract_json(text)
    if not intent:
        raise ValueError("no-json")
    rvv = l1_intent_to_rvv(intent) if level == "L1" else l2_intent_to_rvv(intent, op)
    return rvv, intent


def direct_from_evidence(model_id: str, evidence: str, level: str, mode: str, use_cache: bool) -> tuple[str, dict]:
    if mode == "repair":
        prompt = (DIRECT_REPAIR_L1 if level == "L1" else DIRECT_REPAIR_L2).format(evidence=evidence)
    else:
        prompt = (DIRECT_SOURCEFREE_L1 if level == "L1" else DIRECT_SOURCEFREE_L2).format(evidence=evidence)
    text = LB.chat(model_id, prompt, max_tokens=3600, timeout=220, use_cache=use_cache)
    code = LB.extract_c(text)
    includes = "" if "riscv_vector.h" in code else "#include <riscv_vector.h>\n"
    return includes + code, {"direct": mode}


def save_case(label: str, rvv: str, op, level: str):
    local = DIFF / "kernels" / label
    local.mkdir(parents=True, exist_ok=True)
    (local / "rvv.c").write_text(macros(op, level) + rvv + '\n#include "harness.h"\n'
                                 if "#define DT" not in rvv else rvv, newline="\n")
    (local / "scalar.c").write_text(scalar_kernel(op, level), newline="\n")
    (local / "spec.json").write_text(json.dumps(spec_for(op, level, label)), newline="\n")
    for name in ("rvv.c", "scalar.c", "spec.json"):
        put("server", str(local / name), f"{REMOTE_ROOT}/kernels/{label}/{name}")


def iter_cases(probe: str, ops_l1, ops_l2, models, faults, views):
    if probe in ("all", "repair"):
        for level, ops, registry in (("L1", ops_l1, L1.OPS), ("L2", ops_l2, L2.OPS)):
            for opname in ops:
                op = registry[opname]
                clean = valid_rvv(op, level)
                for fault in faults:
                    evidence = inject_fault(clean, fault)
                    for model in models:
                        yield {
                            "probe": "noisy_repair",
                            "level": level,
                            "operator": opname,
                            "variant": fault,
                            "model": model,
                            "evidence": evidence,
                            "op": op,
                        }
    if probe in ("all", "sourcefree"):
        for level, ops, registry in (("L1", ops_l1, L1.OPS), ("L2", ops_l2, L2.OPS)):
            for opname in ops:
                op = registry[opname]
                for view in views:
                    evidence = source_free_evidence(op, level, view)
                    for model in models:
                        yield {
                            "probe": "source_free",
                            "level": level,
                            "operator": opname,
                            "variant": view,
                            "model": model,
                            "evidence": evidence,
                            "op": op,
                        }


def summarize(tasks, results, seeds):
    planned_per_kernel = seeds * 5 * len(EFFECTIVE_VLENS)
    summary = {}
    for probe in sorted({t["probe"] for t in tasks}):
        summary[probe] = {}
        for arm in ("direct", "agentvec"):
            items = [t for t in tasks if t["probe"] == probe and t["arm"] == arm]
            gen = [t for t in items if t["status"] == "generated"]
            gen_fail = [t for t in items if t["status"] != "generated"]
            build = wrong = full = passed = executed = 0
            for t in gen:
                r = results.get(t["label"], {})
                if r.get("build_error"):
                    build += 1
                    continue
                total = int(r.get("total", 0))
                ok = int(r.get("passed", 0))
                executed += total
                passed += ok
                if total and ok == total:
                    full += 1
                else:
                    wrong += 1
            planned = len(items) * planned_per_kernel
            summary[probe][arm] = {
                "attempted_generated_kernels": len(items),
                "generated_source_files": len(gen),
                "generation_failures": len(gen_fail),
                "build_failures": build,
                "wrong_or_partial": wrong,
                "fully_passing_kernels": full,
                "planned_dynamic_checks": planned,
                "executed_dynamic_checks": executed,
                "passed_dynamic_checks": passed,
                "planned_check_spr": round(passed / planned, 4) if planned else 0.0,
            }
    by_level = {}
    for level in ("L1", "L2"):
        by_level[level] = {}
        for probe in sorted({t["probe"] for t in tasks}):
            by_level[level][probe] = {}
            for arm in ("direct", "agentvec"):
                items = [t for t in tasks if t["level"] == level and t["probe"] == probe and t["arm"] == arm]
                planned = len(items) * planned_per_kernel
                passed = sum(int(results.get(t["label"], {}).get("passed", 0)) for t in items if t["status"] == "generated")
                by_level[level][probe][arm] = {
                    "planned_dynamic_checks": planned,
                    "passed_dynamic_checks": passed,
                    "planned_check_spr": round(passed / planned, 4) if planned else 0.0,
                }
    return {"by_probe": summary, "by_level": by_level}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["all", "repair", "sourcefree"], default="all")
    ap.add_argument("--l1-ops", default=",".join(L1_OPS))
    ap.add_argument("--l2-ops", default=",".join(L2_OPS))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--faults", default=",".join(FAULTS))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--arms", default="direct,agentvec")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", default=str(HERE / "_s04_repair_sourcefree.json"))
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--no-clean-remote", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    models = [m for m in args.models.split(",") if m]
    arms = [a for a in args.arms.split(",") if a]
    ops_l1 = [o for o in args.l1_ops.split(",") if o]
    ops_l2 = [o for o in args.l2_ops.split(",") if o]
    faults = [f for f in args.faults.split(",") if f]
    views = [v for v in args.views.split(",") if v]

    if not args.no_clean_remote:
        if not REMOTE_ROOT.startswith("/tmp/agentvec_s04_"):
            raise RuntimeError(f"refusing to clean non-temp remote root: {REMOTE_ROOT}")
        run("server", f"rm -rf {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}/kernels", timeout=60)
    put("server", str(DIFF / "runner.py"), f"{REMOTE_ROOT}/runner.py")
    put("server", str(DIFF / "harness.h"), f"{REMOTE_ROOT}/harness.h")

    base_cases = list(iter_cases(args.probe, ops_l1, ops_l2, models, faults, views))
    tasks = []
    total = len(base_cases) * len(arms)
    idx = 0
    for case in base_cases:
        for arm in arms:
            idx += 1
            model = case["model"]
            label = "s04_{probe}_{arm}_{level}_{op}_{model}_{variant}".format(
                probe=case["probe"],
                arm=arm,
                level=case["level"].lower(),
                op=case["operator"],
                model=smodel(model),
                variant=case["variant"],
            )
            rec = {k: v for k, v in case.items() if k not in ("evidence", "op")}
            rec.update({"arm": arm, "label": label, "status": "pending"})
            print(f"[{idx}/{total}] {label}", flush=True)
            try:
                mid = LB.MODELS[model]
                if arm == "agentvec":
                    mode = "repair" if case["probe"] == "noisy_repair" else "sourcefree"
                    rvv, note = agentvec_from_evidence(mid, case["evidence"], case["op"], case["level"], mode, args.use_cache)
                else:
                    mode = "repair" if case["probe"] == "noisy_repair" else "sourcefree"
                    rvv, note = direct_from_evidence(mid, case["evidence"], case["level"], mode, args.use_cache)
                save_case(label, rvv, case["op"], case["level"])
                rec.update(status="generated", note=note)
                print("  -> generated", flush=True)
            except Exception as e:
                rec.update(status="generation_failure", error=str(e)[:240])
                print(f"  -> generation_failure: {rec['error']}", flush=True)
            tasks.append(rec)

    print("running S04 differential tests on configured RVV target...", flush=True)
    vstr = ",".join(map(str, EFFECTIVE_VLENS))
    rc, out, err = run(
        "server",
        f"cd {REMOTE_ROOT} && python3 runner.py --root {REMOTE_ROOT} --gcc {GCC} --qemu {QEMU} "
        f"--vlens {vstr} --seeds {args.seeds} 2>&1",
        timeout=14400,
    )
    i = out.find("{")
    report = json.loads(out[i:]) if i >= 0 else {
        "parse_error": {"stdout": out[-2000:], "stderr": err[-2000:], "rc": rc}
    }
    results = report.get("kernels", {})
    final = {
        "config": {
            "probe": args.probe,
            "l1_ops": ops_l1,
            "l2_ops": ops_l2,
            "models": models,
            "faults": faults,
            "views": views,
            "arms": arms,
            "seeds": args.seeds,
            "vlens": list(EFFECTIVE_VLENS),
            "remote_root": REMOTE_ROOT,
            "gcc": GCC,
            "qemu": QEMU,
        },
        "tasks": tasks,
        "runner_report": report,
        "summary": summarize(tasks, results, args.seeds),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(final["summary"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
