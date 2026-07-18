"""P1-1 baseline mini-study (reviewer request): head-to-head on reduction kernels,
five method categories, judged by the SAME differential-testing harness on the server
(riscv-gcc + qemu across VLEN) plus an objdump vector-instruction count (did it actually
vectorize?). Deterministic arms (gcc autovec, vectrans-style fast-math pragma, AgentVec
lowering) need no model; the direct-RVV arm is a frontier proposer (Opus-4.8, in-loop,
as in the paper). This is a prompt-level reproduction of each method *category*, not the
original authors' artifacts. Run: python -u run_baseline_mini.py"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, HOSTS
from lowering import emit_reduce_kernel

GCC = HOSTS["server"]["riscv_gcc"]; QEMU = HOSTS["server"]["qemu"]
OBJD = GCC.rsplit("-", 1)[0] + "-objdump"
R = "/path/to/agentvec_lab/difftest"
DIFF = os.path.join(os.path.dirname(__file__), "..", "lab", "difftest")
MAC = "#include <math.h>\n#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"

ORACLE = {
    "dot":  "    float s=0; for(int i=0;i<n;i++) s+=a[i]*b[i]; out[0]=s;",
    "asum": "    (void)b; float s=0; for(int i=0;i<n;i++) s+=fabsf(a[i]); out[0]=s;",
    "nrm2": "    (void)b; float s=0; for(int i=0;i<n;i++) s+=a[i]*a[i]; out[0]=sqrtf(s);",
}
AVP = {"dot": ("sum", "ab", None), "asum": ("sum", "abs_a", None), "nrm2": ("sum", "aa", "sqrt")}


def wrap(body):
    return (MAC + "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
            "    (void)c;\n" + body + "\n}\n#include \"harness.h\"\n")


# direct-RVV (frontier proposer writes RVV intrinsics directly == pure-LLM / IntrinTrans category)
DIRECT = {
"dot": """#include <riscv_vector.h>
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)c; size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax); size_t vl;
    for (int i = 0; i < n; i += (int)vl) { vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        acc = __riscv_vfmacc_vv_f32m1(acc, __riscv_vle32_v_f32m1(a + i, vl), __riscv_vle32_v_f32m1(b + i, vl), vl); }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(r);
}""",
"asum": """#include <riscv_vector.h>
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c; size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax); size_t vl;
    for (int i = 0; i < n; i += (int)vl) { vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        acc = __riscv_vfadd_vv_f32m1(acc, __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(a + i, vl), vl), vl); }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    out[0] = __riscv_vfmv_f_s_f32m1_f32(r);
}""",
"nrm2": """#include <riscv_vector.h>
void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {
    (void)b; (void)c; size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax); size_t vl;
    for (int i = 0; i < n; i += (int)vl) { vl = __riscv_vsetvl_e32m1((size_t)(n - i));
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, va, va, vl); }
    vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    r = __riscv_vfredusum_vs_f32m1_f32m1(acc, r, vlmax);
    out[0] = sqrtf(__riscv_vfmv_f_s_f32m1_f32(r));
}""",
}

NRM2_IDIOM = wrap("""    (void)b; float scale=0.0f, ssq=1.0f;
    for (int i=0;i<n;i++){ float x=a[i]; if (x!=0.0f){ float ax=fabsf(x);
        if (scale<ax){ float r=scale/ax; ssq=1.0f+ssq*r*r; scale=ax; }
        else { float r=ax/scale; ssq+=r*r; } } }
    out[0] = (scale==0.0f)?0.0f: scale*sqrtf(ssq);""")


def candidate(kernel, arm):
    if arm == "gcc_autovec":
        return wrap(ORACLE[kernel])
    if arm == "vectrans":
        return (MAC + '#pragma GCC optimize("fast-math,tree-vectorize")\n'
                "void agentvec_kernel(const DT *a, const DT *b, const DT *c, DT *out, int n) {\n"
                "    (void)c;\n" + ORACLE[kernel] + "\n}\n#include \"harness.h\"\n")
    if arm == "agentvec":
        return emit_reduce_kernel(*AVP[kernel])
    if arm == "direct_rvv":
        return MAC + DIRECT[kernel] + '\n#include "harness.h"\n'
    if arm == "idiom":
        return NRM2_IDIOM
    raise ValueError(arm)


def oracle_file(kernel):
    return wrap(ORACLE[kernel])


def save(label, rvv, kernel):
    d = os.path.join(DIFF, "kernels", label); os.makedirs(d, exist_ok=True)
    open(f"{d}/rvv.c", "w", newline="\n").write(rvv)
    open(f"{d}/scalar.c", "w", newline="\n").write(oracle_file(kernel))
    json.dump({"name": label, "dtype": "f32", "reduce": True, "tier": "float_ulp",
               "rtol": 1e-4, "sizes": [7, 64, 1000, 4096]}, open(f"{d}/spec.json", "w"))
    for f in ("rvv.c", "scalar.c", "spec.json"):
        put("server", f"{d}/{f}", f"{R}/kernels/{label}/{f}")


ARMS = ["gcc_autovec", "vectrans", "direct_rvv", "agentvec"]
KERNELS = ["dot", "asum", "nrm2"]


def main():
    labels = []
    for k in KERNELS:
        for arm in ARMS:
            lbl = f"mini_{arm}_{k}"; save(lbl, candidate(k, arm), k); labels.append(lbl)
            print(f"built {lbl}", flush=True)
    save("mini_idiom_nrm2", candidate("nrm2", "idiom"), "nrm2"); labels.append("mini_idiom_nrm2")
    print("built mini_idiom_nrm2", flush=True)

    put("server", os.path.join(DIFF, "runner.py"), f"{R}/runner.py")
    print("=== difftest across VLEN {128,256,512} ===", flush=True)
    rc, out, err = run("server", f"cd {R} && python3 runner.py --root {R} --gcc {GCC} --qemu {QEMU} "
                       f"--seeds 8 --only {','.join(labels)} 2>&1", timeout=4000)
    i = out.find("{"); res = json.loads(out[i:])["kernels"] if i >= 0 else {}

    # vector-instruction count per candidate (objdump of the built rvv binary)
    vcount = {}
    for lbl in labels:
        cmd = (f"{OBJD} -d {R}/build/{lbl}_rvv 2>/dev/null | "
               f"awk '/<agentvec_kernel>:/{{f=1;next}} /^$/{{f=0}} f' | "
               f"grep -cE '\\sv[a-z]'")
        rc2, o2, _ = run("server", cmd)
        vcount[lbl] = o2.strip()

    print("\n================ BASELINE MINI-STUDY RESULTS ================", flush=True)
    print(f"{'kernel':6} {'arm':12} {'compile':8} {'SPR(VLA)':9} {'#vec-instr':10}", flush=True)
    def row(k, arm, lbl):
        v = res.get(lbl, {})
        comp = "FAIL" if v.get("build_error") else "ok"
        spr = "n/a" if v.get("build_error") else v.get("spr")
        print(f"{k:6} {arm:12} {comp:8} {str(spr):9} {vcount.get(lbl,'?'):>10}", flush=True)
    for k in KERNELS:
        for arm in ARMS:
            row(k, arm, f"mini_{arm}_{k}")
    row("nrm2", "idiom", "mini_idiom_nrm2")
    json.dump({"diff": res, "vcount": vcount}, open("_baseline_mini.json", "w"), indent=2)
    print("\nsaved _baseline_mini.json", flush=True)


if __name__ == "__main__":
    main()
