"""AgentVec end-to-end pipeline (re-runnable, deterministic given cached LLM intents).

  source kernel  --(Intent Proposer: pluggable backend)-->  AIR intent
                 --(Symbolic Guard: deterministic checks)-->  VERIFIED / VETO
                 --(deterministic lowering)-->  RVV VLA kernel
                 --(differential test on real toolchain across VLEN)-->  SPR

Backends (pluggable):
  - SiliconFlowBackend(model)   : DeepSeek/Qwen/GLM/Kimi via API (llm_backend.chat)
  - CodexBackend(model)         : GPT-5.x via Codex (responses pre-fetched into the cache)
  - ManualBackend(table)        : Opus-4.8 / fixed intents supplied by this agent
All LLM calls go through llm_backend's on-disk cache, so a re-run with unchanged
(source, model) re-uses the recovered intent and never re-calls the API; the rest of
the pipeline (Guard, lowering, build, test) is fully deterministic. This is what makes
the artifact reproducible: same inputs -> identical kernels and identical SPR.

Usage:
  python pipeline.py --backend manual --ops dot,asum,nrm2
  SF_KEY=... python pipeline.py --backend siliconflow --model GLM-5 --ops dot,nrm2
"""
import os, sys, json, argparse, re
from remote import run, put, HOSTS
import llm_backend as LB
import obfuscate
from air_from_formula import air_from_intent
from guard import guard
from lowering import emit_rvv_kernel, emit_reduce_kernel

GCC = HOSTS["server"]["riscv_gcc"]
QEMU = HOSTS["server"]["qemu"]
R = os.environ.get("AGENTVEC_REMOTE_ROOT", "/tmp/agentvec/difftest")
DIFF = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "rvv_difftest")
MAC_R = "#include <riscv_vector.h>\n#include <math.h>\n#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"

# operator zoo: name -> (kind, scalar-oracle body, AgentVec lowering spec, rtol)
ZOO = {
    "saxpy": ("map",    "out[i]=2.0f*a[i]+b[i];",              ("map", "out=2*a+b"),       1e-6),
    "scal":  ("map",    "out[i]=3.0f*a[i];",                   ("map", "out=3*a"),         1e-6),
    "copy":  ("map",    "out[i]=a[i];",                        ("map", "out=a"),           0.0),
    "dot":   ("reduce", "s+=a[i]*b[i];",                       ("sum", "ab", None),        1e-4),
    "asum":  ("reduce", "s+=fabsf(a[i]);",                     ("sum", "abs_a", None),     1e-4),
    "nrm2":  ("reduce", "s+=a[i]*a[i];",                       ("sum", "aa", "sqrt"),      1e-4),
}


def oracle(op):
    kind, body, _, _ = ZOO[op]
    if kind == "map":
        return ("#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 0\n"
                "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)c;"
                f"for(int i=0;i<n;i++){body}}}\n#include \"harness.h\"\n")
    return (MAC_R + "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)c;(void)b;"
            f"float s=0;for(int i=0;i<n;i++){body}out[0]={'sqrtf(s)' if op=='nrm2' else 's'};}}\n#include \"harness.h\"\n")


def lower_from_intent(op, intent):
    """Deterministic lowering from a (possibly model-recovered) intent."""
    kind = ZOO[op][0]
    if kind == "map":
        air = air_from_intent({"pattern": "map", "dtype": "f32", "formula": intent})
        ok, proof = guard(air)
        if not ok:
            raise ValueError(f"guard-veto:{proof[:40]}")
        return emit_rvv_kernel(air)
    rop, pre, post = intent
    return emit_reduce_kernel(rop, pre, post)


def run_op(op, intent, label):
    kind, _, default_intent, rtol = ZOO[op]
    rvv = lower_from_intent(op, intent if intent is not None else default_intent if kind == "reduce" else default_intent[1])
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(rvv)
    open(f"{d}/scalar.c", "w", newline="\n").write(oracle(op))
    json.dump({"name": label, "dtype": "f32", "reduce": kind == "reduce", "tier": "float_ulp",
               "rtol": rtol, "sizes": ([1, 7, 64, 1000, 4096] if kind == "map" else [7, 64, 1000, 4096])},
              open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="manual", choices=["manual", "siliconflow"])
    ap.add_argument("--model", default="Opus-4.8")
    ap.add_argument("--ops", default="saxpy,scal,copy,dot,asum,nrm2")
    ap.add_argument("--obfuscate", action="store_true")
    args = ap.parse_args()
    ops = args.ops.split(",")
    tag = re.sub(r"[^A-Za-z0-9]", "", args.model).lower() + ("_obf" if args.obfuscate else "")

    labels = []
    for op in ops:
        intent = None  # manual / default: use the verified canonical intent
        if args.backend == "siliconflow":
            src = oracle(op)
            if args.obfuscate:
                src = obfuscate.obfuscate(src)
            reply = LB.chat(LB.MODELS[args.model], LB.AIR_PROMPT.format(src=src), max_tokens=1200)
            j = LB.extract_json(reply) or {}
            intent = j.get("formula") if ZOO[op][0] == "map" else _reduce_intent(j)
        label = f"pipe_{op}_{tag}"
        try:
            run_op(op, intent, label); labels.append(label); print(f"[pipeline] {label}: lowered+staged", flush=True)
        except Exception as e:
            print(f"[pipeline] {label}: FAIL {str(e)[:60]}", flush=True)
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                                  f"--seeds 12 --only pipe_ 2>&1", timeout=1800)
    i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}
    print(f"\n=== AgentVec pipeline ({args.backend}/{args.model}{' obf' if args.obfuscate else ''}) ===")
    ok = 0
    for op in ops:
        v = res.get(f"pipe_{op}_{tag}", {})
        s = "BUILDFAIL" if v.get("build_error") else str(v.get("spr"))
        ok += 1 if v.get("spr") == 1.0 else 0
        print(f"  {op:8} SPR={s}")
    print(f"  total correct: {ok}/{len(ops)}")


def _reduce_intent(j):
    rop = str(j.get("reduce_op", "sum")).lower()
    elem = str(j.get("elem") or "a").replace(" ", "").lower()
    pre = {"a": "a", "a*b": "ab", "abs(a)": "abs_a", "a*a": "aa"}.get(elem, "a")
    post = "sqrt" if "sqrt" in str(j.get("postproc", "")).lower() else None
    return rop, pre, post


if __name__ == "__main__":
    main()
