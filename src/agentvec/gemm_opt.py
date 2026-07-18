"""L3 GEMM performance recovery on the real K1 board (SpacemiT X60).

Combines the two halves the existing kernels each missed:
  - frontier 'tiled'/opus/gpt: register-tiled (C in vregs across k) but NO cache
    blocking -> collapse out-of-cache (0.1 GFLOP/s @ N=512);
  - autotune 'blocked'/rank1: cache-blocked but rank-1 (reload+store C every k)
    -> memory-bound, best 1.096 GFLOP/s (43% of expert).

This generator emits a kernel that does BOTH: a fixed MR-row register-tiled
microkernel (accumulators kept in vector registers over the full Kc loop) wrapped
in MC/KC/NC cache blocking with A/B packing into contiguous, cache-resident,
unit-stride buffers. Tuned to real X60 params (VLEN=256, L1D=32K, L2=512K,
line=64B, fp64 FMA is the binding resource at ~1 fp64/cycle).

Determinism preserved: a fixed (MR,LMUL,MC,KC,NC) yields a fixed kernel; the loop
only SELECTS the config from on-hardware feedback. [i/N] progress prints.
"""
import os, sys, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))
from ssh_helper import run, put, get

DST = "/data/lyw/operator_extract"
GCC = f"{DST}/toolchain/riscv-gcc-14.3/bin/riscv64-linux-gcc"
QEMU = f"{DST}/toolchain/qemu-riscv64/qemu-riscv64"
WORK = "/data/lyw/agentvec_lab/gemm_l3"
BDIR = "/home/dgc/agentvec_lab/gemm_opt"
MARCH = "rv64gcv_zvl256b_zba_zbb_zbc_zbs"
EXPERT_FALLBACK = 2.545  # OpenBLAS dgemm GFLOP/s @ N=512 (re-measured below if possible)


def gen_kernel(MR=8, L=1, MC=128, KC=256, NC=256, pack=True):
    """Register-tiled (MR rows, VLA columns, LMUL=L) + MC/KC/NC blocking + packing."""
    T = f"f64m{L}"
    rows = range(MR)
    # microkernel body, fully unrolled over MR rows so accumulators stay in registers
    cptr = "\n            ".join(
        f"double*cr{r}=&C[(size_t)(ic+ir+{r})*N+jc+j];" for r in rows)
    cload = "\n            ".join(
        f"vfloat64m{L}_t c{r}=__riscv_vle64_v_{T}(cr{r},vl);" for r in rows)
    aptr = "\n            ".join(
        f"const double*ap{r}=&Ap[(size_t)(ir+{r})*kc];" for r in rows)
    fmacc = "\n              ".join(
        f"c{r}=__riscv_vfmacc_vf_{T}(c{r},ap{r}[k],vb,vl);" for r in rows)
    cstore = "\n            ".join(
        f"__riscv_vse64_v_{T}(cr{r},c{r},vl);" for r in rows)

    if pack:
        bload = f"vfloat64m{L}_t vb=__riscv_vle64_v_{T}(&Bp[(size_t)k*nc+j],vl);"
        packB = ("for(int k=0;k<kc;k++){const double*s=&B[(size_t)(kc0+k)*N+jc];"
                 "double*d=&Bp[(size_t)k*nc];for(int j=0;j<nc;j++)d[j]=s[j];}")
        packA = ("for(int r=0;r<mc;r++){const double*s=&A[(size_t)(ic+r)*N+kc0];"
                 "double*d=&Ap[(size_t)r*kc];for(int k=0;k<kc;k++)d[k]=s[k];}")
        apref = "ap"  # ap{r}[k] from packed A
        decls = (f"static double Ap[{MC}*{KC}] __attribute__((aligned(64)));\n"
                 f"static double Bp[{KC}*{NC}] __attribute__((aligned(64)));")
    else:
        # no packing: read A/B directly (strided B); ap{r} points into A, vb from B
        bload = f"vfloat64m{L}_t vb=__riscv_vle64_v_{T}(&B[(size_t)(kc0+k)*N+jc+j],vl);"
        packB = ""
        packA = ""
        aptr = "\n            ".join(
            f"const double*ap{r}=&A[(size_t)(ic+ir+{r})*N+kc0];" for r in rows)
        decls = ""

    return f"""#include <riscv_vector.h>
#include <stddef.h>
#define MR {MR}
#define MC {MC}
#define KC {KC}
#define NC {NC}
{decls}
void gemm(int N,const double*A,const double*B,double*C){{
  for(size_t t=0;t<(size_t)N*N;t++)C[t]=0.0;
  for(int jc=0;jc<N;jc+=NC){{ int nc=(jc+NC<N)?NC:(N-jc);
    for(int kc0=0;kc0<N;kc0+=KC){{ int kc=(kc0+KC<N)?KC:(N-kc0);
      {packB}
      for(int ic=0;ic<N;ic+=MC){{ int mc=(ic+MC<N)?MC:(N-ic);
        {packA}
        int ir=0;
        for(;ir+MR<=mc;ir+=MR){{
          for(int j=0;j<nc;){{
            size_t vl=__riscv_vsetvl_e64m{L}((size_t)(nc-j));
            {cptr}
            {cload}
            {aptr}
            for(int k=0;k<kc;k++){{
              {bload}
              {fmacc}
            }}
            {cstore}
            j+=(int)vl;
          }}
        }}
        for(;ir<mc;ir++){{
          for(int j=0;j<nc;){{
            size_t vl=__riscv_vsetvl_e64m{L}((size_t)(nc-j));
            double*cr=&C[(size_t)(ic+ir)*N+jc+j];
            vfloat64m{L}_t c=__riscv_vle64_v_{T}(cr,vl);
            const double*ar={"&Ap[(size_t)ir*kc]" if pack else "&A[(size_t)(ic+ir)*N+kc0]"};
            for(int k=0;k<kc;k++){{
              vfloat64m{L}_t vb={"__riscv_vle64_v_"+T+"(&Bp[(size_t)k*nc+j],vl)" if pack else "__riscv_vle64_v_"+T+"(&B[(size_t)(kc0+k)*N+jc+j],vl)"};
              c=__riscv_vfmacc_vf_{T}(c,ar[k],vb,vl);
            }}
            __riscv_vse64_v_{T}(cr,c,vl);
            j+=(int)vl;
          }}
        }}
      }}
    }}
  }}
}}
"""


def gen_kernel_mt(MR=6, L=4, MC=126, KC=512, NC=256):
    """Multi-thread version of the winning packed register-tiled kernel.
    Bp (B-panel) is packed once per (jc,kc) and shared read-only; the row-panel
    (ic) loop is OpenMP-parallel with a per-thread malloc'd Ap, so threads write
    disjoint C row-blocks (no races). Same microkernel as gen_kernel."""
    T = f"f64m{L}"
    rows = range(MR)
    cptr = "\n            ".join(f"double*cr{r}=&C[(size_t)(ic+ir+{r})*N+jc+j];" for r in rows)
    cload = "\n            ".join(f"vfloat64m{L}_t c{r}=__riscv_vle64_v_{T}(cr{r},vl);" for r in rows)
    aptr = "\n            ".join(f"const double*ap{r}=&Ap[(size_t)(ir+{r})*kc];" for r in rows)
    fmacc = "\n              ".join(f"c{r}=__riscv_vfmacc_vf_{T}(c{r},ap{r}[k],vb,vl);" for r in rows)
    cstore = "\n            ".join(f"__riscv_vse64_v_{T}(cr{r},c{r},vl);" for r in rows)
    return f"""#include <riscv_vector.h>
#include <stddef.h>
#include <stdlib.h>
#define MR {MR}
#define MC {MC}
#define KC {KC}
#define NC {NC}
static double Bp[{KC}*{NC}] __attribute__((aligned(64)));
void gemm(int N,const double*A,const double*B,double*C){{
  for(size_t t=0;t<(size_t)N*N;t++)C[t]=0.0;
  for(int jc=0;jc<N;jc+=NC){{ int nc=(jc+NC<N)?NC:(N-jc);
    for(int kc0=0;kc0<N;kc0+=KC){{ int kc=(kc0+KC<N)?KC:(N-kc0);
      for(int k=0;k<kc;k++){{const double*s=&B[(size_t)(kc0+k)*N+jc];double*d=&Bp[(size_t)k*nc];for(int j=0;j<nc;j++)d[j]=s[j];}}
      #pragma omp parallel for schedule(static)
      for(int ic=0;ic<N;ic+=MC){{ int mc=(ic+MC<N)?MC:(N-ic);
        double*Ap=(double*)malloc((size_t)mc*kc*sizeof(double));
        for(int r=0;r<mc;r++){{const double*s=&A[(size_t)(ic+r)*N+kc0];double*d=&Ap[(size_t)r*kc];for(int k=0;k<kc;k++)d[k]=s[k];}}
        int ir=0;
        for(;ir+MR<=mc;ir+=MR){{
          for(int j=0;j<nc;){{
            size_t vl=__riscv_vsetvl_e64m{L}((size_t)(nc-j));
            {cptr}
            {cload}
            {aptr}
            for(int k=0;k<kc;k++){{
              vfloat64m{L}_t vb=__riscv_vle64_v_{T}(&Bp[(size_t)k*nc+j],vl);
              {fmacc}
            }}
            {cstore}
            j+=(int)vl;
          }}
        }}
        for(;ir<mc;ir++){{
          for(int j=0;j<nc;){{
            size_t vl=__riscv_vsetvl_e64m{L}((size_t)(nc-j));
            double*cr=&C[(size_t)(ic+ir)*N+jc+j];
            vfloat64m{L}_t c=__riscv_vle64_v_{T}(cr,vl);
            const double*ar=&Ap[(size_t)ir*kc];
            for(int k=0;k<kc;k++){{
              vfloat64m{L}_t vb=__riscv_vle64_v_{T}(&Bp[(size_t)k*nc+j],vl);
              c=__riscv_vfmacc_vf_{T}(c,ar[k],vb,vl);
            }}
            __riscv_vse64_v_{T}(cr,c,vl);
            j+=(int)vl;
          }}
        }}
        free(Ap);
      }}
    }}
  }}
}}
"""


def gen_kernel_mnk(MR=6, L=4, MC=126, KC=512, NC=256, transB=False):
    """General rectangular GEMM C(MxN) = A(MxK) * op(B), packed register-tiled.
    Same winning schedule as gen_kernel, generalized to independent M,N,K. With
    transB=True it computes C = A * B^T (B is N x K, row-major) -- the access pattern
    of SYRK (C = A A^T) -- by transposing during the pack, so the microkernel is
    unchanged. Signature: void gemm(int M,int N,int K,const double*A,const double*B,double*C)."""
    T = f"f64m{L}"
    rows = range(MR)
    cptr = "\n            ".join(f"double*cr{r}=&C[(size_t)(ic+ir+{r})*N+jc+j];" for r in rows)
    cload = "\n            ".join(f"vfloat64m{L}_t c{r}=__riscv_vle64_v_{T}(cr{r},vl);" for r in rows)
    aptr = "\n            ".join(f"const double*ap{r}=&Ap[(size_t)(ir+{r})*kc];" for r in rows)
    fmacc = "\n              ".join(f"c{r}=__riscv_vfmacc_vf_{T}(c{r},ap{r}[k],vb,vl);" for r in rows)
    cstore = "\n            ".join(f"__riscv_vse64_v_{T}(cr{r},c{r},vl);" for r in rows)
    # pack B block (kc x nc) from B; transB reads B[(jc+j) ,(kc0+k)] (B is N x K)
    if transB:
        packB = ("for(int k=0;k<kc;k++){double*d=&Bp[(size_t)k*nc];"
                 "for(int j=0;j<nc;j++)d[j]=B[(size_t)(jc+j)*K+(kc0+k)];}")
    else:
        packB = ("for(int k=0;k<kc;k++){const double*s=&B[(size_t)(kc0+k)*N+jc];"
                 "double*d=&Bp[(size_t)k*nc];for(int j=0;j<nc;j++)d[j]=s[j];}")
    packA = ("for(int r=0;r<mc;r++){const double*s=&A[(size_t)(ic+r)*K+kc0];"
             "double*d=&Ap[(size_t)r*kc];for(int k=0;k<kc;k++)d[k]=s[k];}")
    return f"""#include <riscv_vector.h>
#include <stddef.h>
#define MR {MR}
#define MC {MC}
#define KC {KC}
#define NC {NC}
static double Ap[{MC}*{KC}] __attribute__((aligned(64)));
static double Bp[{KC}*{NC}] __attribute__((aligned(64)));
void gemm(int M,int N,int K,const double*A,const double*B,double*C){{
  for(size_t t=0;t<(size_t)M*N;t++)C[t]=0.0;
  for(int jc=0;jc<N;jc+=NC){{ int nc=(jc+NC<N)?NC:(N-jc);
    for(int kc0=0;kc0<K;kc0+=KC){{ int kc=(kc0+KC<K)?KC:(K-kc0);
      {packB}
      for(int ic=0;ic<M;ic+=MC){{ int mc=(ic+MC<M)?MC:(M-ic);
        {packA}
        int ir=0;
        for(;ir+MR<=mc;ir+=MR){{
          for(int j=0;j<nc;){{
            size_t vl=__riscv_vsetvl_e64m{L}((size_t)(nc-j));
            {cptr}
            {cload}
            {aptr}
            for(int k=0;k<kc;k++){{
              vfloat64m{L}_t vb=__riscv_vle64_v_{T}(&Bp[(size_t)k*nc+j],vl);
              {fmacc}
            }}
            {cstore}
            j+=(int)vl;
          }}
        }}
        for(;ir<mc;ir++){{
          for(int j=0;j<nc;){{
            size_t vl=__riscv_vsetvl_e64m{L}((size_t)(nc-j));
            double*cr=&C[(size_t)(ic+ir)*N+jc+j];
            vfloat64m{L}_t c=__riscv_vle64_v_{T}(cr,vl);
            const double*ar=&Ap[(size_t)ir*kc];
            for(int k=0;k<kc;k++){{
              vfloat64m{L}_t vb=__riscv_vle64_v_{T}(&Bp[(size_t)k*nc+j],vl);
              c=__riscv_vfmacc_vf_{T}(c,ar[k],vb,vl);
            }}
            __riscv_vse64_v_{T}(cr,c,vl);
            j+=(int)vl;
          }}
        }}
      }}
    }}
  }}
}}
"""


if __name__ == "__main__":
    # quick local dump to eyeball a candidate
    print(gen_kernel(MR=8, L=1, MC=128, KC=256, NC=256, pack=True))
