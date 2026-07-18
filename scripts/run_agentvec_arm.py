"""AgentVec arm: each model recovers only the INTENT (formula) -> Guard verifies ->
deterministic lowering -> RVV VLA -> differential-test. Contrast with run_pure_llm.py:
here the model never writes RVV; correctness/VLA-scalability are guaranteed by lowering,
so the kernel is correct iff the model identified the right formula. Robust across models."""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
import llm_backend as LB
import obfuscate
from air_from_formula import air_from_intent
from guard import guard
from lowering import emit_rvv_kernel

GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]
R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
ORACLE = open(os.path.join(DIFF, "kernels/saxpy/scalar.c")).read()
SRCS = {"clean": ORACLE, "obf": obfuscate.obfuscate(ORACLE)}


def make(label, model_id, src):
    reply = LB.chat(model_id, LB.AIR_PROMPT.format(src=src), max_tokens=1500, timeout=120)
    intent = LB.extract_json(reply)
    if not intent:
        return None, "no-json"
    air = air_from_intent(intent)               # parse recovered formula -> AIR
    ok, proof = guard(air)                       # deterministic Guard
    if not ok:
        return None, f"guard-veto:{proof[:40]}"
    rvv = emit_rvv_kernel(air)                    # deterministic lowering
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "rvv.c"), "w", newline="\n").write(rvv)
    open(os.path.join(d, "scalar.c"), "w", newline="\n").write(ORACLE)
    json.dump({"name": label, "dtype": "f32", "reduce": False, "tier": "float_ulp",
               "rtol": 1e-6, "sizes": [1, 7, 64, 1000, 4096]}, open(os.path.join(d, "spec.json"), "w"))
    return d, intent.get("formula", "?")


def main():
    meta = {}
    for short, mid in LB.MODELS.items():
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        for variant, src in SRCS.items():
            label = f"av_{sm}_{variant}"
            try:
                d, info = make(label, mid, src)
                if d:
                    for f in ("rvv.c", "scalar.c", "spec.json"):
                        put("server", os.path.join(d, f), f"{R}/kernels/{label}/{f}")
                    meta[label] = ("ok", info); print(f"{label}: intent='{info}'")
                else:
                    meta[label] = ("fail", info); print(f"{label}: {info}")
            except Exception as e:
                meta[label] = ("err", str(e)[:60]); print(f"{label}: ERR {type(e).__name__} {str(e)[:80]}")
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                                  f"--seeds 15 --only av_ 2>&1", timeout=1200)
    i = out.find("{"); d = json.loads(out[i:]) if i >= 0 else {"kernels": {}}
    print(f"\n{'model':14}{'clean SPR':>12}{'obf SPR':>12}")
    for short in LB.MODELS:
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        def fmt(label):
            if meta.get(label, ("",))[0] != "ok": return meta.get(label, ("?",""))[1][:10]
            v = d["kernels"].get(label, {})
            return "BUILD_FAIL" if v.get("build_error") else str(v.get("spr"))
        print(f"{short:14}{fmt(f'av_{sm}_clean'):>12}{fmt(f'av_{sm}_obf'):>12}")


if __name__ == "__main__":
    main()
