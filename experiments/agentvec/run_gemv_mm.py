"""L2 gemv for the multi-model figure: verify AgentVec gemv (deterministic) correct
across VLEN, then each open model writes pure-LLM RVV gemv and we test it."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
import llm_backend as LB
DST = "/path/to/operator_extract"; GCC = f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"
QEMU = f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"; work = "/path/to/agentvec_lab/gemv_l2"
PROMPT = """Write a RISC-V Vector (RVV 1.0) GEMV kernel in C. Compute y = A*x for a row-major
square N x N matrix A, vectors x,y of length N, type double. Signature EXACTLY:
  void gemv(int N, const double *A, const double *x, double *y);
Vector-length-agnostic: use __riscv_vsetvl_e64m1 strip-mining, no hardcoded lane count; correct for ANY N.
Output ONLY the function in one ```c code block with the <riscv_vector.h> include. No explanation."""
MODELS = {k: LB.MODELS[k] for k in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")}


def test(tag, src_remote):
    b = run("server", f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c {src_remote} -o gv_{tag}.out 2>&1 && echo OK", timeout=180)[1]
    if "OK" not in b: return "buildfail"
    for v in (128, 256, 512):
        o = run("server", f"cd {work} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 gv_{tag}.out 2>&1 | grep SUMMARY", timeout=180)[1]
        if not re.search(r"fails=0/15", o): return "FAIL"
    return "pass"


def main():
    run("server", f"mkdir -p {work}")
    for f in ("driver.c", "agentvec.c"): put("server", f"gemv_l2/{f}", f"{work}/{f}")
    print("AgentVec gemv (deterministic):", test("av", "agentvec.c"), flush=True)
    res = {}
    for short, mid in MODELS.items():
        try:
            code = LB.extract_c(LB.chat(mid, PROMPT, max_tokens=2200, timeout=120))
        except Exception as e:
            print(f"{short:14} GEN_FAIL", flush=True); res[short] = "genfail"; continue
        if "riscv_vector.h" not in code: code = "#include <riscv_vector.h>\n" + code
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        local = f"_gen/gemv_{sm}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(code)
        put("server", local, f"{work}/gemv_{sm}.c")
        r = test(sm, f"gemv_{sm}.c"); res[short] = r
        print(f"{short:14} pure-LLM gemv: {r}", flush=True)
    import json; json.dump(res, open("_gemv_mm.json", "w"))


if __name__ == "__main__":
    main()
