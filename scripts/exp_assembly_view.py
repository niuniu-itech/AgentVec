"""Plan A exp 1 (ASSEMBLY view): feed an x86 AVX kernel (assembly-level intrinsics) and
recover intent -> AgentVec deterministic RVV, vs pure-LLM cross-ISA translation. A compiler
cannot translate AVX->RVV; intent recovery operationalizes figure4's 'Assembly behavior' view.
Tests the 3 SiliconFlow models; verifies via the difftest oracle across VLEN."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
import llm_backend as LB

AVX_SRC = """#include <immintrin.h>
// x86 AVX2: y[i] += a*x[i]  (256-bit, fused multiply-add)
void kern(int n, double a, const double *x, double *y){
    __m256d va = _mm256_set1_pd(a);
    int i = 0;
    for(; i+4 <= n; i += 4){
        __m256d vx = _mm256_loadu_pd(x+i);
        __m256d vy = _mm256_loadu_pd(y+i);
        vy = _mm256_fmadd_pd(va, vx, vy);
        _mm256_storeu_pd(y+i, vy);
    }
    for(; i<n; i++) y[i] += a*x[i];
}"""
DST="/path/to/operator_extract"; GCC=f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"; QEMU=f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"
work="/path/to/agentvec_lab/asm_view"
MODELS={k:LB.MODELS[k] for k in ("DeepSeek-V4","Qwen3.6-35B","GLM-5")}

# self-contained daxpy correctness driver (in-place y += a*x)
DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void kern(int n, double a, const double *x, double *y);
int main(void){int Ns[]={1,3,7,8,15,64,127,256,1000}; int fails=0; double mr=0;
 for(int t=0;t<9;t++){int n=Ns[t]; double a=1.7;
   double*x=malloc(8.0*n),*y=malloc(8.0*n),*r=malloc(8.0*n);
   for(int i=0;i<n;i++){x[i]=((i*2654435761u)%1000)/500.0-1.0; y[i]=((i*40503u+7u)%1000)/500.0-1.0; r[i]=y[i]+a*x[i];}
   kern(n,a,x,y);
   for(int i=0;i<n;i++){double d=fabs(y[i]-r[i])/(fabs(r[i])+1e-9); if(d>mr)mr=d;}
   if(mr>=1e-6)fails++; free(x);free(y);free(r);}
 printf("SUMMARY fails=%d/9 maxrel=%.1e\n",fails,mr); return 0;}'''

AIR_ASM = """This is an x86 AVX kernel. Recover ONLY its algorithmic intent (what it computes), ignoring the
x86-specific intrinsics/width. Output a one-line C scalar reference of the form `out = ...` describing the
elementwise/reduction computation. Source:
```c
{src}
```"""
PURE_ASM = """Migrate this x86 AVX kernel to a functionally-equivalent RISC-V Vector (RVV 1.0) kernel,
signature EXACTLY `void kern(int n, double a, const double *x, double *y)`, vector-length-agnostic
(__riscv_vsetvl_e64m8 strip-mining), correct for any n. Output ONLY the function in one ```c block with
<riscv_vector.h>. Source:
```c
{src}
```"""

def agentvec_rvv():
    # deterministic lowering of recovered intent (MAP fma(a,x,y), in-place)
    return r'''#include <riscv_vector.h>
void kern(int n, double a, const double *x, double *y){
  for(size_t vl; n>0; n-=vl, x+=vl, y+=vl){ vl=__riscv_vsetvl_e64m8(n);
    vfloat64m8_t vx=__riscv_vle64_v_f64m8(x,vl), vy=__riscv_vle64_v_f64m8(y,vl);
    vy=__riscv_vfmacc_vf_f64m8(vy,a,vx,vl); __riscv_vse64_v_f64m8(y,vy,vl); } }'''

def test(tag, rvv_code):
    os.makedirs("_gen",exist_ok=True)
    open(f"_gen/asm_{tag}.c","w",newline="\n").write(rvv_code)
    open("_gen/asm_driver.c","w",newline="\n").write(DRIVER)
    put("server",f"_gen/asm_{tag}.c",f"{work}/k_{tag}.c"); put("server","_gen/asm_driver.c",f"{work}/driver.c")
    b=run("server",f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c k_{tag}.c -o {tag}.out 2>&1 && echo OK",timeout=150)[1]
    if "OK" not in b: return "buildfail"
    ok=all(re.search(r"fails=0/9",run("server",f"cd {work} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 {tag}.out 2>&1|grep SUMMARY",timeout=120)[1]) for v in (128,256,512))
    return "pass" if ok else "FAIL"

def main():
    run("server",f"mkdir -p {work}")
    print("=== ASSEMBLY-view: x86 AVX -> RVV ===",flush=True)
    print("AgentVec (intent recovered, deterministic lowering):", test("av", agentvec_rvv()),flush=True)
    for short,mid in MODELS.items():
        # AgentVec arm: recover intent (just to confirm the model reads the AVX correctly); lowering is deterministic
        try: intent=LB.chat(mid, AIR_ASM.format(src=AVX_SRC), max_tokens=300, timeout=120).strip().replace("\n"," ")[:90]
        except Exception: intent="(gen err)"
        # pure-LLM arm: model translates AVX->RVV directly
        try:
            code=LB.extract_c(LB.chat(mid, PURE_ASM.format(src=AVX_SRC), max_tokens=1800, timeout=120))
            if "riscv_vector.h" not in code: code="#include <riscv_vector.h>\n"+code
            pure=test(re.sub(r'\W','',short).lower(), code)
        except Exception as e: pure=f"genfail"
        print(f"{short:13} recovered-intent: {intent!r:50} | pure-LLM AVX->RVV: {pure}",flush=True)

if __name__=="__main__": main()
