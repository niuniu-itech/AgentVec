"""Second composite operator: layernorm = a 5-node dataflow DAG
  t1=contract(+,x); mean=t1/n;  t2=contract(+,(x-mean)^2); var=t2/n;  y=map[(x-mean)*rsqrt(var+eps)]
Two reductions + a map, chained through scalar mean/var. Uses sqrt (once), not a per-element
transcendental, so qemu is fast and we verify at full sizes. AgentVec recovers the composite intent
and lowers to a fused 3-pass RVV kernel; pure-LLM writes RVV layernorm directly. 3 SiliconFlow models."""
import os, sys, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put
import llm_backend as LB
DST="/path/to/operator_extract"; GCC=f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"; QEMU=f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"
work="/path/to/agentvec_lab/layernorm"
MODELS={k:LB.MODELS[k] for k in ("DeepSeek-V4","Qwen3.6-35B","GLM-5")}

SCALAR_SRC = """/* layer normalization over a length-n float vector (eps=1e-5) */
void layernorm(int n, const float *x, float *y){
    float s=0; for(int i=0;i<n;i++) s+=x[i];  float mean=s/n;
    float v=0; for(int i=0;i<n;i++){ float d=x[i]-mean; v+=d*d; }  float var=v/n;
    float rstd=1.0f/sqrtf(var+1e-5f);
    for(int i=0;i<n;i++) y[i]=(x[i]-mean)*rstd;
}"""

AGENTVEC = r'''#include <riscv_vector.h>
#include <math.h>
void layernorm(int n, const float *x, float *y){
  if(n<=0) return; size_t vlmax=__riscv_vsetvlmax_e32m8();
  /* stage 1: s = sum x ; mean = s/n   (contract +) */
  vfloat32m8_t vs=__riscv_vfmv_v_f_f32m8(0.0f,vlmax);
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i)); vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); vs=__riscv_vfadd_vv_f32m8(vs,v,vl); i+=(int)vl;}
  vfloat32m1_t r0=__riscv_vfmv_v_f_f32m1(0.0f,1); r0=__riscv_vfredusum_vs_f32m8_f32m1(vs,r0,vlmax);
  float mean=__riscv_vfmv_f_s_f32m1_f32(r0)/(float)n;
  /* stage 2: var = (1/n) sum (x-mean)^2   (map sub,sq ; contract +) */
  vfloat32m8_t va=__riscv_vfmv_v_f_f32m8(0.0f,vlmax);
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i));
    vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); v=__riscv_vfsub_vf_f32m8(v,mean,vl);
    va=__riscv_vfmacc_vv_f32m8(va,v,v,vl); i+=(int)vl;}
  vfloat32m1_t r1=__riscv_vfmv_v_f_f32m1(0.0f,1); r1=__riscv_vfredusum_vs_f32m8_f32m1(va,r1,vlmax);
  float var=__riscv_vfmv_f_s_f32m1_f32(r1)/(float)n;
  float rstd=1.0f/sqrtf(var+1e-5f);
  /* stage 3: y = (x-mean)*rstd   (map) */
  for(int i=0;i<n;){size_t vl=__riscv_vsetvl_e32m8((size_t)(n-i));
    vfloat32m8_t v=__riscv_vle32_v_f32m8(x+i,vl); v=__riscv_vfsub_vf_f32m8(v,mean,vl); v=__riscv_vfmul_vf_f32m8(v,rstd,vl);
    __riscv_vse32_v_f32m8(y+i,v,vl); i+=(int)vl;}
}'''

DRIVER = r'''#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void layernorm(int n, const float *x, float *y);
static void ref(int n, const float *x, float *y){
  float s=0; for(int i=0;i<n;i++)s+=x[i]; float mean=s/n;
  float v=0; for(int i=0;i<n;i++){float d=x[i]-mean; v+=d*d;} float var=v/n;
  float rstd=1.0f/sqrtf(var+1e-5f);
  for(int i=0;i<n;i++) y[i]=(x[i]-mean)*rstd;
}
int main(void){int Ns[]={7,16,33,64,255,1000}; int fails=0; double mr=0;
  for(int t=0;t<6;t++){int n=Ns[t];
    float*x=malloc(4.0*n),*y=malloc(4.0*n),*r=malloc(4.0*n);
    for(int i=0;i<n;i++) x[i]=((i*2654435761u)%1000)/100.0f-5.0f;
    ref(n,x,r); layernorm(n,x,y);
    double sm=0; for(int i=0;i<n;i++){double d=fabs((double)y[i]-r[i])/(fabs(r[i])+1e-6); if(d>sm)sm=d;}
    if(sm>=1e-3)fails++; if(sm>mr)mr=sm; free(x);free(y);free(r);}
  printf("SUMMARY fails=%d/6 maxrel=%.1e\n",fails,mr); return 0;}'''

AIR_C = """Recover ONLY the algorithmic intent of this kernel as a DATAFLOW PIPELINE of stages (each stage
is map/reduce over an element op; later stages may read earlier outputs). Output a short numbered list
`t1=...; t2=...; ...`. Source:
```c
{src}
```"""
PURE = """Migrate this C kernel to a RISC-V Vector (RVV 1.0) kernel, signature EXACTLY
`void layernorm(int n, const float *x, float *y)`, vector-length-agnostic (__riscv_vsetvl_e32m8),
correct for any n. Output ONLY the function in one ```c block with <riscv_vector.h> and <math.h>. Source:
```c
{src}
```"""

def test(tag, code):
    os.makedirs("_gen",exist_ok=True); open(f"_gen/ln_{tag}.c","w",newline="\n").write(code); open("_gen/ln_driver.c","w",newline="\n").write(DRIVER)
    put("server",f"_gen/ln_{tag}.c",f"{work}/k_{tag}.c"); put("server","_gen/ln_driver.c",f"{work}/driver.c")
    b=run("server",f"cd {work} && {GCC} -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver.c k_{tag}.c -lm -o {tag}.out 2>&1 && echo OK",timeout=150)[1]
    if "OK" not in b: return "buildfail"
    ok=all(re.search(r"fails=0/6",run("server",f"cd {work} && {QEMU} -cpu rv64,v=true,vlen={v},vext_spec=v1.0 {tag}.out 2>&1|grep SUMMARY",timeout=150)[1]) for v in (128,256,512))
    return "pass" if ok else "FAIL"

def main():
    run("server",f"mkdir -p {work}")
    print("=== COMPOSITE operator 2: layernorm (5-node DAG) ===",flush=True)
    print("AgentVec (composite intent -> deterministic fused RVV):", test("av", AGENTVEC),flush=True)
    for short,mid in MODELS.items():
        try: intent=LB.chat(mid, AIR_C.format(src=SCALAR_SRC), max_tokens=400, timeout=120).strip().replace("\n"," ")[:110]
        except Exception: intent="(gen err)"
        try:
            code=LB.extract_c(LB.chat(mid, PURE.format(src=SCALAR_SRC), max_tokens=2200, timeout=150))
            for h in ("riscv_vector.h","math.h"):
                if h not in code: code=f"#include <{h}>\n"+code
            pure=test(re.sub(r'\W','',short).lower(), code)
        except Exception: pure="genfail"
        print(f"{short:13} composite-intent: {intent!r:72} | pure-LLM RVV layernorm: {pure}",flush=True)

if __name__=="__main__": main()
