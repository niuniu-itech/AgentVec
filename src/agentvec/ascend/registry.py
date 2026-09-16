"""Phase 1 — the Intent Proposer (the "propose" of the propose->contract->arbitrate
discipline). It reads a cuBLAS operator and recovers its hardware-independent
algorithmic intent into the AIR schema. As in AgentVec, the model only fills declared
schema slots. These registered contracts provide the independent comparison
for supplied or model-proposed intent. Lowered kernels still require compilation
and differential validation on the target.

Each entry pairs the cuBLAS "source" (what is being migrated) with the recovered AIR.
"""
from .air import (AIR, SAlgo, CPhy, MMap, VMeta, Expr,
                 Pattern, Dep, Alias, Access, Layout, Status)


def _load(s): return Expr("load", src=s)
def _mulc(a, c): return Expr("mulc", val=c, args=[a])     # placeholder alpha=c at intent time
def _fma_axpy(): return Expr("add", args=[Expr("mulc", val=None, args=[_load("x")]), _load("y")])

# cuBLAS source signatures (single precision) being migrated, for the report narrative.
CUBLAS_SRC = {
    "saxpy": "cublasSaxpy(handle,n,&alpha,x,1,y,1);  // y[i] = alpha*x[i] + y[i]",
    "sscal": "cublasSscal(handle,n,&alpha,x,1);      // x[i] = alpha*x[i]",
    "scopy": "cublasScopy(handle,n,x,1,y,1);         // y[i] = x[i]",
    "sdot":  "cublasSdot (handle,n,x,1,y,1,&res);    // res = sum_i x[i]*y[i]",
    "sasum": "cublasSasum(handle,n,x,1,&res);        // res = sum_i |x[i]|",
    "snrm2": "cublasSnrm2(handle,n,x,1,&res);        // res = sqrt(sum_i x[i]^2)",
    "sgemv": "cublasSgemv(handle,N,m,n,&alpha,A,lda,x,1,&beta,y,1); // y = alpha*A*x + beta*y",
    "sgemm": "cublasSgemm(handle,N,N,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc); // C = alpha*A*B + beta*C",
    "sgemm_nt": "cublasSgemm(handle,N,T,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc); // C = alpha*A*B^T + beta*C",
    "ssyrk": "cublasSsyrk(handle,UPLO,N,n,k,&alpha,A,lda,&beta,C,ldc);        // C = alpha*A*A^T + beta*C",
    "strsv": "cublasStrsv(handle,UPLO,N,DIAG,n,A,lda,x,1);          // solve A*x = b (triangular)",
    "strsm": "cublasStrsm(handle,L,UPLO,N,DIAG,m,n,&alpha,A,lda,B,ldb);       // solve A*X = B (multi-RHS)",
}


def recover(op: str) -> AIR:
    """Recover the AIR intent for a cuBLAS operator (Track-alpha proposal + Track-beta
    static facts of C_phy). DRAFT status — the Guard (V-AIR) verifies it next."""
    f32 = lambda **kw: CPhy(dtype="f32", **kw)

    if op == "saxpy":   # y = alpha*x + y  : MAP, fma(alpha,x,y)
        return AIR(name=op, s_algo=SAlgo(Pattern.MAP, elem_op=_fma_axpy(), has_alpha=True, has_beta=False),
                   c_phy=f32(dep=Dep.NONE, alias=Alias.DISJOINT), m_map=MMap(Access.UNIT_STRIDE, bound="n"))
    if op == "sscal":   # x = alpha*x      : MAP in-place, mulc(alpha,x)
        return AIR(name=op, s_algo=SAlgo(Pattern.MAP, elem_op=Expr("mulc", val=None, args=[_load("x")]),
                                         has_alpha=True, has_beta=False),
                   c_phy=f32(dep=Dep.NONE, alias=Alias.DISJOINT), m_map=MMap(Access.UNIT_STRIDE, bound="n"))
    if op == "scopy":   # y = x            : MAP, load(x)
        return AIR(name=op, s_algo=SAlgo(Pattern.MAP, elem_op=_load("x"), has_alpha=False, has_beta=False),
                   c_phy=f32(dep=Dep.NONE, alias=Alias.DISJOINT), m_map=MMap(Access.UNIT_STRIDE, bound="n"))
    if op == "sdot":    # sum x*y          : REDUCE sum, pre=ab
        return AIR(name=op, s_algo=SAlgo(Pattern.REDUCE, reduce_op="sum", reduce_pre="ab"),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"), m_map=MMap(Access.UNIT_STRIDE, bound="n"))
    if op == "sasum":   # sum |x|          : REDUCE sum, pre=abs_a
        return AIR(name=op, s_algo=SAlgo(Pattern.REDUCE, reduce_op="sum", reduce_pre="abs_a"),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"), m_map=MMap(Access.UNIT_STRIDE, bound="n"))
    if op == "snrm2":   # sqrt(sum x^2)    : REDUCE sum, pre=aa, post=sqrt
        return AIR(name=op, s_algo=SAlgo(Pattern.REDUCE, reduce_op="sum", reduce_pre="aa", reduce_post="sqrt"),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"), m_map=MMap(Access.UNIT_STRIDE, bound="n"))
    if op == "sgemv":   # y = alpha*A*x + beta*y : GEMV  (map_i o contract_j)
        return AIR(name=op, s_algo=SAlgo(Pattern.GEMV, has_alpha=True, has_beta=True),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"),
                   m_map=MMap(Access.UNIT_STRIDE, layout=Layout.ROW_MAJOR, bound="m,n"))
    if op == "sgemm":   # C = alpha*A*B + beta*C : GEMM
        return AIR(name=op, s_algo=SAlgo(Pattern.GEMM, has_alpha=True, has_beta=True),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"),
                   m_map=MMap(Access.UNIT_STRIDE, layout=Layout.ROW_MAJOR, bound="m,n,k"))
    if op == "sgemm_nt":  # C = alpha*A*B^T + beta*C : GEMM-NT (row-dot form)
        return AIR(name=op, s_algo=SAlgo(Pattern.GEMM_NT, trans_b=True, has_alpha=True, has_beta=True),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"),
                   m_map=MMap(Access.UNIT_STRIDE, layout=Layout.ROW_MAJOR, bound="m,n,k"))
    if op == "ssyrk":   # C = alpha*A*A^T + beta*C : symmetric rank-k
        return AIR(name=op, s_algo=SAlgo(Pattern.SYRK, trans_b=True, has_alpha=True, has_beta=True),
                   c_phy=f32(dep=Dep.REDUCTION, equiv="ulp"),
                   m_map=MMap(Access.UNIT_STRIDE, layout=Layout.ROW_MAJOR, bound="m,k"))
    if op == "strsm":   # triangular solve, multiple RHS : loop-carried -> VETO
        return AIR(name=op, s_algo=SAlgo(Pattern.TRSM, has_alpha=True, has_beta=False),
                   c_phy=f32(dep=Dep.LOOP_CARRIED, dep_distance=1, equiv="exact"),
                   m_map=MMap(Access.STRIDED, layout=Layout.ROW_MAJOR, bound="m,n"))
    if op == "strsv":   # triangular solve : forward substitution reads x_{<i} -> loop-carried
        return AIR(name=op, s_algo=SAlgo(Pattern.TRSV, has_alpha=False, has_beta=False),
                   c_phy=f32(dep=Dep.LOOP_CARRIED, dep_distance=1, equiv="exact"),
                   m_map=MMap(Access.STRIDED, layout=Layout.ROW_MAJOR, bound="n"))
    raise ValueError(f"unknown cuBLAS op {op}")


ALL_OPS = ["saxpy", "sscal", "scopy", "sdot", "sasum", "snrm2", "sgemv",
           "sgemm", "sgemm_nt", "ssyrk", "strsv", "strsm"]


if __name__ == "__main__":
    for op in ALL_OPS:
        air = recover(op)
        ok, msg = air.validate_schema()
        print(f"{op:7} pattern={air.s_algo.pattern.value:6} schema_ok={ok} ({msg})")
