"""Open-model pure-LLM GEMM (L3) for the multi-model figure. Each SiliconFlow model
writes an RVV GEMM; we test correctness across VLEN with the L3 driver. AgentVec arm
is the deterministic lowering (correct by construction), so it is not re-queried."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get, HOSTS
import llm_backend as LB
DST = "/path/to/operator_extract"; GCC = f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"
QEMU = f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"; work = "/path/to/agentvec_lab/gemm_l3"
PROMPT = """Write a RISC-V Vector (RVV 1.0) GEMM kernel in C. Compute C = A*B for row-major square
N x N matrices, type double. Signature EXACTLY:
  void gemm(int N, const double *A, const double *B, double *C);
Vector-length-agnostic: use __riscv_vsetvl_e64m1 strip-mining, no hardcoded lane count; correct for ANY N.
Output ONLY the function in one ```c code block with the <riscv_vector.h> include. No explanation."""
MODELS = {k: LB.MODELS[k] for k in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")}


def main():
    print(f"{'model':14}{'pure-LLM GEMM (correct across VLEN)':>36}", flush=True)
    res = {}
    for short, mid in MODELS.items():
        try:
            code = LB.extract_c(LB.chat(mid, PROMPT, max_tokens=2200, timeout=120))
        except Exception as e:
            print(f"{short:14} GEN_FAIL {str(e)[:40]}", flush=True); res[short] = "genfail"; continue
        if "riscv_vector.h" not in code:
            code = "#include <riscv_vector.h>\n" + code
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        local = f"_gen/gemm_{sm}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(code)
        put("server", local, f"{work}/gemm_{sm}.c")
        b = run("server", f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c gemm_{sm}.c -o gemm_{sm}.out 2>&1 && echo OK", timeout=180)[1]
        if "OK" not in b:
            print(f"{short:14}{'BUILDFAIL':>36}", flush=True); res[short] = "buildfail"; continue
        allok = True
        for v in (128, 256, 512):
            o = run("server", f"cd {work} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 gemm_{sm}.out 2>&1 | grep SUMMARY", timeout=180)[1]
            if not re.search(r"fails=0/15", o): allok = False
        res[short] = "pass" if allok else "FAIL"
        print(f"{short:14}{('PASS (all VLEN)' if allok else 'FAIL'):>36}", flush=True)
    import json; json.dump(res, open("_gemm_mm.json", "w"))


if __name__ == "__main__":
    main()
