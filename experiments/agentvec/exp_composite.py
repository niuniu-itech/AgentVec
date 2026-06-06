"""Composite-operator experiment: softmax = a 4-stage dataflow pipeline
  t1 = contract(max, x);  t2 = map(exp(x - t1));  t3 = contract(+, t2);  y = map(t2 / t3)
AgentVec recovers the composite intent (the pipeline/DAG) and deterministically lowers it to a
fused multi-pass RVV kernel (vectorized max-reduce + normalize; scalar exp). Verified vs a scalar
oracle across VLEN. Pure-LLM arm: models write RVV softmax directly. Tests 3 SiliconFlow models."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
import llm_backend as LB
DST="/path/to/operator_extract"; GCC=f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"; QEMU=f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"
work="/path/to/agentvec_lab/composite"
MODELS={k:LB.MODELS[k] for k in ("DeepSeek-V4","Qwen3.6-35B","GLM-5")}

SCALAR_SRC = """/* numerically-stable softmax over a length-n float vector */
void softmax(int n, const float *x, float *y){
    float m = -3.4e38f; for(int i=0;i<n;i++) if(x[i]>m) m=x[i];
    float s = 0.0f;     for(int i=0;i<n;i++){ y[i]=expf(x[i]-m); s+=y[i]; }
    for(int i=0;i<n;i++) y[i] /= s;
}"""

# AgentVec deterministic lowering of the recovered composite intent (fused 3-pass RVV)
AGENTVEC = r'''#include <riscv_vector.h>
#include <math.h>
void softmax(int n, const float *x, float *y){
  if(n<=0) return;
  size_t vlmax=__riscv_vsetvlmax_e32m8();
  /* stage 1: m = max_i x_i   (contract max) */
  vfloat32m8_t vm=__riscv_vfmv_v_f_f32m8(-3.4e38f,vlmax);
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i)); vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); vm=__riscv_vfmax_vv_f32m8(vm,v,vl); i+=(int)vl;}
  vfloat32m1_t rm=__riscv_vfmv_v_f_f32m1(-3.4e38f,1); rm=__riscv_vfredmax_vs_f32m8_f32m1(vm,rm,vlmax);
  float m=__riscv_vfmv_f_s_f32m1_f32(rm);
  /* stage 2: y_i = exp(x_i - m); s = sum y_i   (map exp ; contract +) */
  float s=0.0f; for(int i=0;i<n;i++){ float e=expf(x[i]-m); y[i]=e; s+=e; }
  /* stage 3: y_i *= 1/s   (map) */
  float inv=1.0f/s;
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i)); vfloat32m8_t v=__riscv_vle32_v_f32m8(y+i,vl); v=__riscv_vfmul_vf_f32m8(v,inv,vl); __riscv_vse32_v_f32m8(y+i,v,vl); i+=(int)vl;}
}'''

DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void softmax(int n, const float *x, float *y);
static void ref(int n, const float *x, float *y){
  float m=-3.4e38f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
  float s=0; for(int i=0;i<n;i++){y[i]=expf(x[i]-m); s+=y[i];}
  for(int i=0;i<n;i++) y[i]/=s;
}
int main(void){int Ns[]={7,16,33,64}; int fails=0; double mr=0;
  for(int t=0;t<4;t++){int n=Ns[t];
    float*x=malloc(4.0*n),*y=malloc(4.0*n),*r=malloc(4.0*n);
    for(int i=0;i<n;i++) x[i]=((i*2654435761u)%1000)/100.0f-5.0f;
    ref(n,x,r); softmax(n,x,y);
    double sm=0; for(int i=0;i<n;i++){double d=fabs((double)y[i]-r[i])/(fabs(r[i])+1e-9); if(d>sm)sm=d;}
    if(sm>=1e-4)fails++; if(sm>mr)mr=sm; free(x);free(y);free(r);}
  printf("SUMMARY fails=%d/4 maxrel=%.1e\n",fails,mr); return 0;}'''

AIR_C = """Recover ONLY the algorithmic intent of this kernel as a DATAFLOW PIPELINE of stages
(each stage is map/reduce over an element op; later stages may read earlier outputs). Output a short
numbered list `t1=...; t2=...; ...`. Source:
```c
{src}
```"""
PURE = """Migrate this C kernel to a RISC-V Vector (RVV 1.0) kernel, signature EXACTLY
`void softmax(int n, const float *x, float *y)`, vector-length-agnostic (__riscv_vsetvl_e32m8),
correct for any n. You may use scalar expf() for the exponential. Output ONLY the function in one
```c block with <riscv_vector.h> and <math.h>. Source:
```c
{src}
```"""

def test(tag, code):
    os.makedirs("_gen",exist_ok=True); open(f"_gen/sm_{tag}.c","w",newline="\n").write(code); open("_gen/sm_driver.c","w",newline="\n").write(DRIVER)
    put("server",f"_gen/sm_{tag}.c",f"{work}/k_{tag}.c"); put("server","_gen/sm_driver.c",f"{work}/driver.c")
    b=run("server",f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c k_{tag}.c -lm -o {tag}.out 2>&1 && echo OK",timeout=150)[1]
    if "OK" not in b: return "buildfail"
    ok=all(re.search(r"fails=0/4",run("server",f"cd {work} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 {tag}.out 2>&1|grep SUMMARY",timeout=180)[1]) for v in (128,256,512))
    return "pass" if ok else "FAIL"

def main():
    run("server",f"mkdir -p {work}")
    print("=== COMPOSITE operator: softmax (4-stage pipeline) ===",flush=True)
    print("AgentVec (composite intent -> deterministic fused RVV):", test("av", AGENTVEC),flush=True)
    for short,mid in MODELS.items():
        try: intent=LB.chat(mid, AIR_C.format(src=SCALAR_SRC), max_tokens=400, timeout=120).strip().replace("\n"," ")[:120]
        except Exception: intent="(gen err)"
        try:
            code=LB.extract_c(LB.chat(mid, PURE.format(src=SCALAR_SRC), max_tokens=2200, timeout=150))
            for h in ("riscv_vector.h","math.h"):
                if h not in code: code=f"#include <{h}>\n"+code
            pure=test(re.sub(r'\W','',short).lower(), code)
        except Exception: pure="genfail"
        print(f"{short:13} composite-intent: {intent!r:74} | pure-LLM RVV softmax: {pure}",flush=True)

if __name__=="__main__": main()
