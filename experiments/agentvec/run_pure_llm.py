"""Multi-model PURE-LLM migration study: each model writes an RVV kernel directly
from a (clean or obfuscated) scalar source; we differential-test it across VLEN.
This is the 'pure LLM' arm (no AIR, no Guard). Run: SF_KEY=... python run_pure_llm.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
import llm_backend as LB
import obfuscate

GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]
R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")

MACROS = "#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 0\n"
ORACLE = open(os.path.join(DIFF, "kernels/saxpy/scalar.c")).read()   # ground truth
CLEAN_SRC = ORACLE
OBF_SRC = obfuscate.obfuscate(ORACLE)


def make_kernel(label, model_id, src):
    reply = LB.chat(model_id, LB.PURE_LLM_PROMPT.format(src=src))
    code = LB.extract_c(reply)
    # ensure riscv_vector.h + harness contract
    body = MACROS
    if "riscv_vector.h" not in code:
        body += "#include <riscv_vector.h>\n"
    body += code + '\n#include "harness.h"\n'
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "rvv.c"), "w", newline="\n").write(body)
    open(os.path.join(d, "scalar.c"), "w", newline="\n").write(ORACLE)
    json.dump({"name": label, "dtype": "f32", "reduce": False, "tier": "float_ulp",
               "rtol": 1e-6, "sizes": [1, 7, 64, 1000, 4096]},
              open(os.path.join(d, "spec.json"), "w"))
    return d, len(code)


def main():
    labels = []
    for short, mid in LB.MODELS.items():
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        for variant, src in (("clean", CLEAN_SRC), ("obf", OBF_SRC)):
            label = f"llm_{sm}_{variant}"
            try:
                d, n = make_kernel(label, mid, src)
                for f in ("rvv.c", "scalar.c", "spec.json"):
                    put("server", os.path.join(d, f), f"{R}/kernels/{label}/{f}")
                labels.append(label); print(f"generated {label} ({n} chars)")
            except Exception as e:
                print(f"FAILED gen {label}: {type(e).__name__} {str(e)[:120]}")
    # one difftest pass over just these kernels
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                                  f"--seeds 15 --only llm_ 2>&1", timeout=1800)
    i = out.find("{")
    if i < 0:
        print("RUN ERR:", out[-600:], err[-200:]); return
    d = json.loads(out[i:])
    print(f"\n{'model':14}{'clean SPR':>12}{'obf SPR':>12}")
    for short in LB.MODELS:
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        cv = d["kernels"].get(f"llm_{sm}_clean", {})
        ov = d["kernels"].get(f"llm_{sm}_obf", {})
        def fmt(v):
            if v.get("build_error"): return "BUILD_FAIL"
            return str(v.get("spr"))
        print(f"{short:14}{fmt(cv):>12}{fmt(ov):>12}")


if __name__ == "__main__":
    main()
