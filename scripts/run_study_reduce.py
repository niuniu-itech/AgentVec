"""Multi-model study on REDUCTION ops (dot, asum, nrm2) where pure-LLM must write a
correct vectorized reduction (hard) vs AgentVec where the model only recovers the
reduce intent and deterministic lowering guarantees correctness. Run with python -u."""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
import llm_backend as LB, obfuscate
from lowering import emit_reduce_kernel

GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]; R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
MACROS = "#include <math.h>\n#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"

# op -> (scalar-oracle body, AgentVec lowering params, rtol)
OPS = {
    "dot":  ("    float s=0; for(int i=0;i<n;i++) s+=a[i]*b[i]; out[0]=s;",       ("sum","ab",None),   1e-4),
    "asum": ("    (void)b; float s=0; for(int i=0;i<n;i++) s+=fabsf(a[i]); out[0]=s;", ("sum","abs_a",None), 1e-4),
    "nrm2": ("    (void)b; float s=0; for(int i=0;i<n;i++) s+=a[i]*a[i]; out[0]=sqrtf(s);",("sum","aa","sqrt"), 1e-4),
}


def oracle(op):
    return (MACROS + f"void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {{\n"
            f"    (void)c;\n{OPS[op][0]}\n}}\n#include \"harness.h\"\n")


def save(label, rvv, op):
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(rvv)
    open(f"{d}/scalar.c", "w", newline="\n").write(oracle(op))
    json.dump({"name": label, "dtype": "f32", "reduce": True, "tier": "float_ulp",
               "rtol": OPS[op][2], "sizes": [7, 64, 1000, 4096]}, open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


def gen_pure(mid, src):
    code = LB.extract_c(LB.chat(mid, LB.PURE_LLM_PROMPT.format(src=src), max_tokens=2200, timeout=110))
    return MACROS + ("" if "riscv_vector.h" in code else "#include <riscv_vector.h>\n") + code + '\n#include "harness.h"\n'


def intent_to_reduce(intent):
    """Map the model's recovered reduce intent to lowering params (honest: a wrong
    recovery yields a wrong kernel)."""
    rop = str(intent.get("reduce_op", "sum")).lower()
    elem = str(intent.get("elem") or intent.get("formula") or "a").replace(" ", "").lower()
    elem = elem.split("=")[-1]
    pre = {"a": "a", "a[i]": "a", "a*b": "ab", "b*a": "ab", "a[i]*b[i]": "ab",
           "abs(a)": "abs_a", "fabs(a)": "abs_a", "|a|": "abs_a",
           "a*a": "aa", "a^2": "aa", "a**2": "aa", "a[i]*a[i]": "aa"}.get(elem)
    post = "sqrt" if "sqrt" in str(intent.get("postproc", "")).lower() else None
    return rop, pre, post


def gen_av(mid, src, op):
    intent = LB.extract_json(LB.chat(mid, LB.AIR_PROMPT.format(src=src), max_tokens=1200, timeout=110))
    if not intent:
        raise ValueError("no-json")
    rop, pre, post = intent_to_reduce(intent)    # USE the model's recovered intent
    if pre is None or rop not in ("sum", "max", "min"):
        raise ValueError(f"bad-intent:{intent.get('reduce_op')}/{intent.get('elem')}")
    return emit_reduce_kernel(rop, pre, post)     # deterministic lowering of the recovered intent


def main():
    info = {}
    models = {k: LB.MODELS[k] for k in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")}
    total = len(OPS) * len(models) * 2 * 2
    idx = 0
    for op in OPS:
        base = oracle(op); src_clean = base; src_obf = obfuscate.obfuscate(base)
        for short, mid in models.items():
            sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
            for variant, src in (("clean", src_clean), ("obf", src_obf)):
                for arm in ("llm", "av"):
                    idx += 1
                    label = f"{arm}_{op}_{sm}_{variant}"
                    print(f"[{idx}/{total}] generating {label} ({'pure-LLM RVV' if arm=='llm' else 'AgentVec intent->lowering'})", flush=True)
                    try:
                        rvv = gen_pure(mid, src) if arm == "llm" else gen_av(mid, src, op)
                        save(label, rvv, op); info[label] = "ok"
                        print(f"[{idx}/{total}] {label}: ok", flush=True)
                    except Exception as e:
                        info[label] = f"GEN_FAIL:{str(e)[:30]}"; print(f"[{idx}/{total}] {label}: {info[label]}", flush=True)
    print("testing...", flush=True)
    put("server", os.path.join(DIFF, "runner.py"), f"{R}/runner.py")   # ensure -lm fix is on server
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                                  f"--seeds 10 --only llm_dot,llm_asum,llm_nrm2,av_dot,av_asum,av_nrm2 2>&1", timeout=3000)
    i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}
    def spr(label):
        if info.get(label, "").startswith("GEN_FAIL"): return "genfail"
        v = res.get(label);
        return "n/a" if v is None else ("BUILDFAIL" if v.get("build_error") else str(v.get("spr")))
    for op in OPS:
        print(f"\n== {op} ==   PURE-LLM clean/obf | AgentVec clean/obf", flush=True)
        for short in models:
            sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
            print(f"{short:12} | {spr(f'llm_{op}_{sm}_clean'):>9} / {spr(f'llm_{op}_{sm}_obf'):>9} | "
                  f"{spr(f'av_{op}_{sm}_clean'):>9} / {spr(f'av_{op}_{sm}_obf'):>9}", flush=True)


if __name__ == "__main__":
    main()
