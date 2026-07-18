"""P1-2 ablation: NO-EXAMPLE prompt + HELD-OUT skeletons.

Removes the four worked examples from the AIR prompt (keeping only the bare schema) and
re-tests intent recovery -- including max/min reductions, which NEVER appear in the prompt
examples (only `sum` is shown). This refutes the reviewer's "open-model success is just
low-entropy classification over the listed schema" objection: if recovery survives with no
examples, on skeletons it was never shown, the success is genuine structured recovery.

AgentVec arm only: the model recovers an intent JSON, the deterministic lowering emits the
RVV kernel, and the independent differential tester returns SPR across VLEN {128,256,512}.
Run:  SF_KEY=$(cat ../../SF_KEY.txt) python -u run_noexample_ablation.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
import llm_backend as LB
from lowering import emit_reduce_kernel

GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]; R = "/data/lyw/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
MACROS = "#include <math.h>\n#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"

# NO-EXAMPLE prompt: keep the schema spec, drop the four worked examples
NOEX = LB.AIR_PROMPT.split("Examples:")[0].rstrip() + "\nJSON only."

OPS = {  # op -> (scalar oracle body, (reduce_op, pre, postproc), shown_in_examples?)
    "dot":  ("    float s=0; for(int i=0;i<n;i++) s+=a[i]*b[i]; out[0]=s;",            ("sum", "ab", None),    True),
    "asum": ("    (void)b; float s=0; for(int i=0;i<n;i++) s+=fabsf(a[i]); out[0]=s;", ("sum", "abs_a", None), True),
    "nrm2": ("    (void)b; float s=0; for(int i=0;i<n;i++) s+=a[i]*a[i]; out[0]=sqrtf(s);", ("sum", "aa", "sqrt"), True),
    "max":  ("    (void)b; float s=a[0]; for(int i=1;i<n;i++) if(a[i]>s) s=a[i]; out[0]=s;", ("max", "a", None), False),
    "min":  ("    (void)b; float s=a[0]; for(int i=1;i<n;i++) if(a[i]<s) s=a[i]; out[0]=s;", ("min", "a", None), False),
}


def oracle(op):
    return (MACROS + "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            f"    (void)c;\n{OPS[op][0]}\n}}\n#include \"harness.h\"\n")


def save(label, rvv, op):
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(rvv)
    open(f"{d}/scalar.c", "w", newline="\n").write(oracle(op))
    json.dump({"name": label, "dtype": "f32", "reduce": True, "tier": "float_ulp",
               "rtol": 1e-4, "sizes": [7, 64, 1000, 4096]}, open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


def intent_to_reduce(intent):
    rop = str(intent.get("reduce_op", "sum")).lower()
    elem = str(intent.get("elem") or intent.get("formula") or "a").replace(" ", "").lower().split("=")[-1]
    pre = {"a": "a", "a[i]": "a", "a*b": "ab", "b*a": "ab", "a[i]*b[i]": "ab",
           "abs(a)": "abs_a", "fabs(a)": "abs_a", "|a|": "abs_a",
           "a*a": "aa", "a^2": "aa", "a**2": "aa", "a[i]*a[i]": "aa"}.get(elem)
    post = "sqrt" if "sqrt" in str(intent.get("postproc", "")).lower() else None
    return rop, pre, post


def main():
    models = {k: LB.MODELS[k] for k in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")}
    info, intents, labels = {}, {}, []
    n = len(OPS) * len(models); idx = 0
    for op in OPS:
        src = oracle(op)
        for short, mid in models.items():
            sm = re.sub(r"[^A-Za-z0-9]", "", short).lower(); idx += 1
            label = f"nx_{op}_{sm}"
            try:
                intent = LB.extract_json(LB.chat(mid, NOEX.format(src=src), max_tokens=1200, timeout=110))
                if not intent:
                    info[label] = "NO_JSON"; print(f"[{idx}/{n}] {label}: NO_JSON", flush=True); continue
                intents[label] = {k: intent.get(k) for k in ("reduce_op", "elem", "postproc")}
                rop, pre, post = intent_to_reduce(intent)
                if pre is None or rop not in ("sum", "max", "min"):
                    info[label] = f"BAD_INTENT:{intent.get('reduce_op')}/{intent.get('elem')}"
                    print(f"[{idx}/{n}] {label}: {info[label]}", flush=True); continue
                save(label, emit_reduce_kernel(rop, pre, post), op)
                info[label] = "ok"; labels.append(label)
                print(f"[{idx}/{n}] {label}: ok  intent={intents[label]}", flush=True)
            except Exception as e:
                info[label] = f"GEN_FAIL:{str(e)[:40]}"; print(f"[{idx}/{n}] {label}: {info[label]}", flush=True)

    print("testing on server (QEMU across VLEN)...", flush=True)
    put("server", os.path.join(DIFF, "runner.py"), f"{R}/runner.py")
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                                  f"--seeds 10 --only {','.join(labels)} 2>&1", timeout=3000)
    i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}

    def spr(label):
        if info.get(label, "").startswith(("GEN_FAIL", "BAD_INTENT", "NO_JSON")):
            return info[label][:9]
        v = res.get(label)
        return "n/a" if v is None else ("BUILDFAIL" if v.get("build_error") else str(v.get("spr")))

    rows = {}
    print("\n== NO-EXAMPLE prompt, AgentVec arm -- SPR  (max,min are held-out skeletons) ==", flush=True)
    print(f"{'model':14} " + " ".join(f"{op:>9}" for op in OPS), flush=True)
    for short in models:
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        rows[short] = {op: spr(f"nx_{op}_{sm}") for op in OPS}
        print(f"{short:14} " + " ".join(f"{rows[short][op]:>9}" for op in OPS), flush=True)
    json.dump({"prompt": "no-example", "arm": "AgentVec", "ops": list(OPS), "held_out": ["max", "min"],
               "rows": rows, "intents": intents, "info": info},
              open("_noexample_ablation.json", "w"), indent=2)
    print("\nwrote _noexample_ablation.json", flush=True)


if __name__ == "__main__":
    main()
