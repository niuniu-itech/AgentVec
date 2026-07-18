"""Add GPT-5.5 (via Codex) and Opus-4.8 (this agent) to the multi-model reduce table.
Pure-LLM arm: each model's directly-written RVV reduction, tested across VLEN.
GPT code is pasted from Codex; Opus code is written by this agent (Opus-4.8)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]; R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
MAC = "#include <riscv_vector.h>\n#include <math.h>\n#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"
ORACLE = {
    "dot":  "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)c;float s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];out[0]=s;}",
    "asum": "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)b;(void)c;float s=0;for(int i=0;i<n;i++)s+=fabsf(a[i]);out[0]=s;}",
    "nrm2": "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)b;(void)c;float s=0;for(int i=0;i<n;i++)s+=a[i]*a[i];out[0]=sqrtf(s);}",
}

# --- GPT-5.5 pure-LLM (verbatim from Codex) ---
GPT = {
"dot": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n) {
    (void)c;
    size_t vlmax = __riscv_vsetvl_e32m1((size_t)-1);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, va, vb, vl);
        i += (int)vl;
    }
    vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(sum);
}''',
"asum": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)b; (void)c;
    if (n <= 0){ out[0]=0.0f; return; }
    size_t vlmax = __riscv_vsetvl_e32m1((size_t)-1);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    int i=0;
    while (i<n){
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t x = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t ax = __riscv_vfabs_v_f32m1(x, vl);
        acc = __riscv_vfadd_vv_f32m1_tu(acc, acc, ax, vl);
        i += (int)vl;
    }
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(sum);
}''',
"nrm2": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)b; (void)c;
    size_t vl = __riscv_vsetvl_e32m1((size_t)n);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (int i = 0; i < n; ) {
        vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(va, va, vl);
        acc = __riscv_vfadd_vv_f32m1(acc, sq, vl);
        i += (int)vl;
    }
    vl = __riscv_vsetvl_e32m1((size_t)n);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t sumv = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl);
    float sum = __riscv_vfmv_f_s_f32m1_f32(sumv);
    out[0] = sqrtf(sum);
}''',
}

# --- Opus-4.8 pure-LLM (written by this agent; uses vse to extract lane 0, since this
#     toolchain lacks __riscv_vfmv_f_s_f32m1) ---
OPUS = {
"dot": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)c;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int i = 0; i < n; ){
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, va, vb, vl);
        i += (int)vl;
    }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    __riscv_vse32_v_f32m1(out, r, 1);
}''',
"asum": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)b; (void)c;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int i = 0; i < n; ){
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(a + i, vl), vl);
        acc = __riscv_vfadd_vv_f32m1(acc, va, vl);
        i += (int)vl;
    }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    __riscv_vse32_v_f32m1(out, r, 1);
}''',
"nrm2": r'''void agentvec_kernel(const float *a, const float *b, const float *c, float *out, int n){
    (void)b; (void)c;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int i = 0; i < n; ){
        size_t vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, va, va, vl);
        i += (int)vl;
    }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    float s; __riscv_vse32_v_f32m1(&s, r, 1);
    out[0] = sqrtf(s);
}''',
}

RTOL = {"dot": 1e-4, "asum": 1e-4, "nrm2": 1e-4}


def save(label, code, op):
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(MAC + code + '\n#include "harness.h"\n')
    open(f"{d}/scalar.c", "w", newline="\n").write(MAC + ORACLE[op] + '\n#include "harness.h"\n')
    json.dump({"name": label, "dtype": "f32", "reduce": True, "tier": "float_ulp",
               "rtol": RTOL[op], "sizes": [7, 64, 1000, 4096]}, open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


labels = []
for op in ("dot", "asum", "nrm2"):
    save(f"gpt_{op}_clean", GPT[op], op); labels.append(f"gpt_{op}_clean")
    save(f"opus_{op}_clean", OPUS[op], op); labels.append(f"opus_{op}_clean")
put("server", os.path.join(DIFF, "runner.py"), f"{R}/runner.py")
rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                              f"--seeds 10 --only gpt_,opus_ 2>&1", timeout=1200)
i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}
def spr(l):
    v = res.get(l); return "n/a" if v is None else ("BUILDFAIL" if v.get("build_error") else str(v.get("spr")))
print(f"\n{'op':8}{'GPT-5.5':>12}{'Opus-4.8':>12}")
for op in ("dot", "asum", "nrm2"):
    print(f"{op:8}{spr(f'gpt_{op}_clean'):>12}{spr(f'opus_{op}_clean'):>12}")
