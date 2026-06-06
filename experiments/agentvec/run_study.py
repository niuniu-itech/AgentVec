"""Unified multi-model study (run with `python -u`): for each model, both arms
(PURE-LLM direct RVV, and AgentVec intent->Guard->lowering) on saxpy clean+obf,
then one cross-VLEN differential-test pass. Robust: per-call timeout, skip on fail."""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
import llm_backend as LB, obfuscate
from air_from_formula import air_from_intent
from guard import guard
from lowering import emit_rvv_kernel

GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]; R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
ORACLE = open(os.path.join(DIFF, "kernels/saxpy/scalar.c")).read()
SRCS = {"clean": ORACLE, "obf": obfuscate.obfuscate(ORACLE)}
MACROS = "#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 0\n"
SPEC = {"dtype": "f32", "reduce": False, "tier": "float_ulp", "rtol": 1e-6, "sizes": [1, 7, 64, 1000, 4096]}


def save(label, rvv):
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(rvv)
    open(f"{d}/scalar.c", "w", newline="\n").write(ORACLE)
    sp = dict(SPEC); sp["name"] = label; json.dump(sp, open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


def gen_pure(mid, src):
    code = LB.extract_c(LB.chat(mid, LB.PURE_LLM_PROMPT.format(src=src), max_tokens=2000, timeout=90))
    body = MACROS + ("" if "riscv_vector.h" in code else "#include <riscv_vector.h>\n") + code + '\n#include "harness.h"\n'
    return body, ("hardcoded-width" if re.search(r"vsetivli|vfmv.*\b[48]\b", code) else "vla")


def gen_av(mid, src):
    intent = LB.extract_json(LB.chat(mid, LB.AIR_PROMPT.format(src=src), max_tokens=1200, timeout=90))
    if not intent: raise ValueError("no-json")
    air = air_from_intent(intent); ok, proof = guard(air)
    if not ok: raise ValueError(f"veto:{proof[:30]}")
    return emit_rvv_kernel(air), intent.get("formula", "?")


def main():
    labels, info = [], {}
    for short, mid in LB.MODELS.items():
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        for variant, src in SRCS.items():
            for arm, fn in (("llm", gen_pure), ("av", gen_av)):
                label = f"{arm}_{sm}_{variant}"
                try:
                    rvv, note = fn(mid, src); save(label, rvv)
                    labels.append(label); info[label] = note
                    print(f"  {label}: {note}", flush=True)
                except Exception as e:
                    info[label] = f"GEN_FAIL:{str(e)[:40]}"
                    print(f"  {label}: {info[label]}", flush=True)
    print("testing across VLEN...", flush=True)
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                                  f"--seeds 12 --only llm_,av_ 2>&1", timeout=2400)
    i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}
    def spr(label):
        if label not in info: return "-"
        if str(info[label]).startswith("GEN_FAIL"): return "genfail"
        v = res.get(label, {})
        return "buildfail" if v.get("build_error") else str(v.get("spr"))
    print(f"\n{'model':12} | PURE-LLM clean/obf | AgentVec clean/obf", flush=True)
    for short in LB.MODELS:
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        print(f"{short:12} |  {spr(f'llm_{sm}_clean'):>5} / {spr(f'llm_{sm}_obf'):>5}   |  "
              f"{spr(f'av_{sm}_clean'):>5} / {spr(f'av_{sm}_obf'):>5}", flush=True)


if __name__ == "__main__":
    main()
