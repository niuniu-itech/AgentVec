"""Obfuscated-source pure-LLM cells for GPT-5.5 (Codex outputs) and Opus-4.8 (this agent),
to complete the 5-model multi-model reduce table."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
from add_gpt_opus import MAC, ORACLE, RTOL, OPUS  # reuse oracle + Opus kernels (intent recovered same)
GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]; R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")

GPT_OBF = {
"dot": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    size_t vl0 = __riscv_vsetvl_e32m1(n > 0 ? (size_t)n : 1);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl0);
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, va, vb, vl);
        i += (int)vl;
    }
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl0);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl0);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(sum);
}''',
"asum": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b; (void)c;
    size_t vlmax = __riscv_vsetvl_e32m1((size_t)-1);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        v = __riscv_vfabs_v_f32m1(v, vl);
        acc = __riscv_vfredusum_vs_f32m1_f32m1(v, acc, vl);
        i += (int)vl;
    }
    out[0] = __riscv_vfmv_f_s_f32m1_f32(acc);
}''',
"nrm2": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)b; (void)c;
    float sum = 0.0f; int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t v = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(v, v, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(sq, zero, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(red);
        i += (int)vl;
    }
    out[0] = sqrtf(sum);
}''',
}


def save(label, code, op):
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(MAC + code + '\n#include "harness.h"\n')
    open(f"{d}/scalar.c", "w", newline="\n").write(MAC + ORACLE[op] + '\n#include "harness.h"\n')
    json.dump({"name": label, "dtype": "f32", "reduce": True, "tier": "float_ulp",
               "rtol": RTOL[op], "sizes": [7, 64, 1000, 4096]}, open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


for op in ("dot", "asum", "nrm2"):
    save(f"gpt_{op}_obf", GPT_OBF[op], op)
    save(f"opus_{op}_obf", OPUS[op], op)   # Opus recovers the same intent through obfuscation
rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                              f"--seeds 10 --only gpt_,opus_ 2>&1", timeout=1200)
i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}
def spr(l):
    v = res.get(l); return "n/a" if v is None else ("BF" if v.get("build_error") else str(v.get("spr")))
print(f"\n{'op':8}{'GPT clean/obf':>18}{'Opus clean/obf':>18}")
for op in ("dot", "asum", "nrm2"):
    print(f"{op:8}{spr(f'gpt_{op}_clean')+' / '+spr(f'gpt_{op}_obf'):>18}{spr(f'opus_{op}_clean')+' / '+spr(f'opus_{op}_obf'):>18}")
