"""Obfuscated-source GEMM (L3) for the obfuscation figure: give each open model an
IM+DCI+T-obfuscated scalar GEMM and ask it to migrate to RVV; test across VLEN.
AgentVec recovers the matmul intent regardless of obfuscation (Delta 0)."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
import llm_backend as LB, obfuscate
DST = "/path/to/operator_extract"; GCC = f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"
QEMU = f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"; work = "/path/to/agentvec_lab/gemm_l3"
SCALAR = """void gemm(int N, const double *A, const double *B, double *C){
  for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double s=0.0;
    for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s; } }"""
MODELS = {k: LB.MODELS[k] for k in ("DeepSeek-V4", "Qwen3.6-35B", "GLM-5")}


def main():
    obf = obfuscate.obfuscate(SCALAR)
    prompt = ("Migrate this C function to a functionally-equivalent RISC-V Vector (RVV 1.0) kernel, "
              "same signature `void gemm(int N, const double *A, const double *B, double *C)`. "
              "Vector-length-agnostic (__riscv_vsetvl_e64m1 strip-mining), correct for any N. "
              "Output ONLY the function in one ```c block with <riscv_vector.h>.\n\n```c\n" + obf + "\n```")
    res = {}
    for short, mid in MODELS.items():
        try:
            code = LB.extract_c(LB.chat(mid, prompt, max_tokens=2200, timeout=120))
        except Exception:
            print(f"{short:14} GEN_FAIL", flush=True); res[short] = "genfail"; continue
        if "riscv_vector.h" not in code: code = "#include <riscv_vector.h>\n" + code
        sm = re.sub(r"[^A-Za-z0-9]", "", short).lower()
        local = f"_gen/gemm_obf_{sm}.c"; os.makedirs("_gen", exist_ok=True); open(local, "w", newline="\n").write(code)
        put("server", local, f"{work}/gemm_obf_{sm}.c")
        b = run("server", f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c gemm_obf_{sm}.c -o gobf_{sm}.out 2>&1 && echo OK", timeout=180)[1]
        if "OK" not in b: print(f"{short:14} obf-gemm: buildfail", flush=True); res[short] = "buildfail"; continue
        ok = all(re.search(r"fails=0/15", run("server", f"cd {work} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 gobf_{sm}.out 2>&1 | grep SUMMARY", timeout=180)[1]) for v in (128, 256, 512))
        res[short] = "pass" if ok else "FAIL"
        print(f"{short:14} obf-gemm pure-LLM: {res[short]}", flush=True)
    import json; json.dump(res, open("_gemm_obf.json", "w"))


if __name__ == "__main__":
    main()
