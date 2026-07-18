"""L1 BLAS lowering: emit expert-signature RVV kernels from a recovered AIR.

Real OpenBLAS L1 kernels are f64, in-place / reduction, strided (inc_x/inc_y),
BLASLONG, with per-op signatures. Each emitter produces a function with the SAME
prototype/name as the operator's `<op>_k_rvv`, so it slots directly into the
migration wiring and is validated by the unmodified benchmark harness.

The (pattern, elem_op) for each kernel is the VERIFIED AIR recovered by the Intent
Proposer from `<op>_k.c`; lowering is deterministic given that AIR.
LMUL=8 (m8), matching the expert kernels.
"""
from dataclasses import dataclass

HEADER = ('#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n'
          '#include <math.h>\n#include <sys/time.h>\n#include <riscv_vector.h>\n'
          '#include "common.h"\n')

# f64 / m8 intrinsic names
S = dict(ct="double", vt="vfloat64m8_t", vt1="vfloat64m1_t",
         setvl="__riscv_vsetvl_e64m8", vle="__riscv_vle64_v_f64m8", vse="__riscv_vse64_v_f64m8",
         vlse="__riscv_vlse64_v_f64m8", vsse="__riscv_vsse64_v_f64m8",
         fmacc_vf="__riscv_vfmacc_vf_f64m8", fmacc_vv="__riscv_vfmacc_vv_f64m8",
         fmul_vf="__riscv_vfmul_vf_f64m8", fadd_vv="__riscv_vfadd_vv_f64m8",
         fabs="__riscv_vfabs_v_f64m8", splat="__riscv_vfmv_v_f_f64m8",
         splat1="__riscv_vfmv_v_f_f64m1", redsum="__riscv_vfredusum_vs_f64m8_f64m1",
         vse1="__riscv_vse64_v_f64m1", setvlmax="__riscv_vsetvlmax_e64m8",
         vfmv_f="__riscv_vfmv_f_s_f64m1",
         fmax_vv="__riscv_vfmax_vv_f64m8", fmin_vv="__riscv_vfmin_vv_f64m8",
         redmax="__riscv_vfredmax_vs_f64m8_f64m1", redmin="__riscv_vfredmin_vs_f64m8_f64m1")


@dataclass
class L1Spec:
    name: str        # must match <op>_k_rvv
    dtype: str       # f64 (d* ops)
    kind: str        # axpy | scal | copy | dot | asum


def emit_axpy(name):  # y[i] += da*x[i]   AIR: MAP, elem_op=fma(da,x,y)
    return HEADER + f"""
int {name}(BLASLONG n, BLASLONG d0, BLASLONG d1, double da,
           double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *d, BLASLONG d2) {{
    if (n <= 0 || da == 0.0) return 0;
    {S['vt']} vx, vy;
    if (inc_x == 1 && inc_y == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {{
            vl = {S['setvl']}(n);
            vx = {S['vle']}(x, vl); vy = {S['vle']}(y, vl);
            vy = {S['fmacc_vf']}(vy, da, vx, vl); {S['vse']}(y, vy, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {{
            vl = {S['setvl']}(n);
            vx = {S['vlse']}(x, sx, vl); vy = {S['vlse']}(y, sy, vl);
            vy = {S['fmacc_vf']}(vy, da, vx, vl); {S['vsse']}(y, sy, vy, vl);
        }}
    }}
    return 0;
}}
"""


def emit_scal(name):  # x[i] = da*x[i]   AIR: MAP in-place, elem_op=mulc(da,x)
    return HEADER + f"""
int {name}(BLASLONG n, BLASLONG d0, BLASLONG d1, double da,
           double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *d, BLASLONG d2) {{
    if (n <= 0) return 0;
    {S['vt']} vx;
    if (inc_x == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl) {{
            vl = {S['setvl']}(n);
            vx = {S['vle']}(x, vl); vx = {S['fmul_vf']}(vx, da, vl); {S['vse']}(x, vx, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {{
            vl = {S['setvl']}(n);
            vx = {S['vlse']}(x, sx, vl); vx = {S['fmul_vf']}(vx, da, vl); {S['vsse']}(x, sx, vx, vl);
        }}
    }}
    return 0;
}}
"""


def emit_copy(name):  # y[i] = x[i]   AIR: MAP, elem_op=load(x)
    return HEADER + f"""
int {name}(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y) {{
    if (n <= 0) return 0;
    {S['vt']} vx;
    if (inc_x == 1 && inc_y == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {{
            vl = {S['setvl']}(n); vx = {S['vle']}(x, vl); {S['vse']}(y, vx, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {{
            vl = {S['setvl']}(n); vx = {S['vlse']}(x, sx, vl); {S['vsse']}(y, sy, vx, vl);
        }}
    }}
    return 0;
}}
"""


def emit_dot(name):  # sum(x[i]*y[i])   AIR: REDUCE(sum) over elem_op=mul(x,y)
    return HEADER + f"""
double {name}(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y) {{
    if (n <= 0) return 0.0;
    size_t vlmax = {S['setvlmax']}();
    {S['vt']} acc = {S['splat']}(0.0, vlmax), vx, vy;
    if (inc_x == 1 && inc_y == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {{
            vl = {S['setvl']}(n);
            vx = {S['vle']}(x, vl); vy = {S['vle']}(y, vl);
            acc = {S['fmacc_vv']}(acc, vx, vy, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {{
            vl = {S['setvl']}(n);
            vx = {S['vlse']}(x, sx, vl); vy = {S['vlse']}(y, sy, vl);
            acc = {S['fmacc_vv']}(acc, vx, vy, vl);
        }}
    }}
    double res;
    {S['vt1']} r = {S['splat1']}(0.0, 1);
    r = {S['redsum']}(acc, r, vlmax);
    {S['vse1']}(&res, r, 1);
    return res;
}}
"""


def emit_asum(name):  # sum(|x[i]|)   AIR: REDUCE(sum) over elem_op=abs(x)
    return HEADER + f"""
double {name}(BLASLONG n, double *x, BLASLONG inc_x) {{
    if (n <= 0 || inc_x <= 0) return 0.0;
    size_t vlmax = {S['setvlmax']}();
    {S['vt']} acc = {S['splat']}(0.0, vlmax), vx;
    if (inc_x == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl) {{
            vl = {S['setvl']}(n); vx = {S['vle']}(x, vl);
            vx = {S['fabs']}(vx, vl); acc = {S['fadd_vv']}(acc, vx, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {{
            vl = {S['setvl']}(n); vx = {S['vlse']}(x, sx, vl);
            vx = {S['fabs']}(vx, vl); acc = {S['fadd_vv']}(acc, vx, vl);
        }}
    }}
    double res;
    {S['vt1']} r = {S['splat1']}(0.0, 1);
    r = {S['redsum']}(acc, r, vlmax);
    {S['vse1']}(&res, r, 1);
    return res;
}}
"""


def emit_nrm2(name):  # sqrt(sum x[i]^2)  AIR: REDUCE(sum) over elem_op=mul(x,x), postproc sqrt
    return HEADER + f"""
double {name}(BLASLONG n, double *x, BLASLONG inc_x) {{
    if (n <= 0 || inc_x <= 0) return 0.0;
    if (n == 1) return fabs(x[0]);
    size_t vlmax = {S['setvlmax']}();
    {S['vt']} acc = {S['splat']}(0.0, vlmax), vx;
    if (inc_x == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl) {{
            vl = {S['setvl']}(n); vx = {S['vle']}(x, vl);
            acc = {S['fmacc_vv']}(acc, vx, vx, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {{
            vl = {S['setvl']}(n); vx = {S['vlse']}(x, sx, vl);
            acc = {S['fmacc_vv']}(acc, vx, vx, vl);
        }}
    }}
    double res; {S['vt1']} r = {S['splat1']}(0.0, 1);
    r = {S['redsum']}(acc, r, vlmax); {S['vse1']}(&res, r, 1);
    return sqrt(res);
}}
"""


def emit_axpby(name):  # y[i] = alpha*x[i] + beta*y[i]   AIR: MAP, elem_op=fma(alpha,x, beta*y)
    s = S
    return HEADER + f"""
int {name}(BLASLONG n, double alpha, double *x, BLASLONG inc_x, double beta, double *y, BLASLONG inc_y) {{
    if (n <= 0) return 0;
    {s['vt']} vx, vy;
    if (inc_x == 1 && inc_y == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {{
            vl = {s['setvl']}(n);
            vx = {s['vle']}(x, vl); vy = {s['vle']}(y, vl);
            vy = {s['fmul_vf']}(vy, beta, vl);
            vy = {s['fmacc_vf']}(vy, alpha, vx, vl);
            {s['vse']}(y, vy, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {{
            vl = {s['setvl']}(n);
            vx = {s['vlse']}(x, sx, vl); vy = {s['vlse']}(y, sy, vl);
            vy = {s['fmul_vf']}(vy, beta, vl); vy = {s['fmacc_vf']}(vy, alpha, vx, vl);
            {s['vsse']}(y, sy, vy, vl);
        }}
    }}
    return 0;
}}
"""


def emit_swap(name):  # swap x[i] <-> y[i]   AIR: MAP two-output
    s = S
    return HEADER + f"""
int {name}(BLASLONG n, BLASLONG d0, BLASLONG d1, double d3,
           double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *d, BLASLONG d2) {{
    if (n <= 0) return 0;
    {s['vt']} vx, vy;
    if (inc_x == 1 && inc_y == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {{
            vl = {s['setvl']}(n);
            vx = {s['vle']}(x, vl); vy = {s['vle']}(y, vl);
            {s['vse']}(x, vy, vl); {s['vse']}(y, vx, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double), sy = inc_y*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {{
            vl = {s['setvl']}(n);
            vx = {s['vlse']}(x, sx, vl); vy = {s['vlse']}(y, sy, vl);
            {s['vsse']}(x, sx, vy, vl); {s['vsse']}(y, sy, vx, vl);
        }}
    }}
    return 0;
}}
"""


def _emit_minmax(name, vv, red, init):
    s = S
    return HEADER + f"""
double {name}(BLASLONG n, double *x, BLASLONG inc_x) {{
    if (n <= 0 || inc_x <= 0) return 0.0;
    size_t vlmax = {s['setvlmax']}();
    {s['vt']} acc = {s['splat']}({init}, vlmax), vx;
    if (inc_x == 1) {{
        for (size_t vl; n > 0; n -= vl, x += vl) {{
            vl = {s['setvl']}(n); vx = {s['vle']}(x, vl); acc = {vv}(acc, vx, vl);
        }}
    }} else {{
        BLASLONG sx = inc_x*(BLASLONG)sizeof(double);
        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {{
            vl = {s['setvl']}(n); vx = {s['vlse']}(x, sx, vl); acc = {vv}(acc, vx, vl);
        }}
    }}
    double res; {s['vt1']} r = {s['splat1']}({init}, 1);
    r = {red}(acc, r, vlmax); {s['vse1']}(&res, r, 1);
    return res;
}}
"""


def emit_max(name): return _emit_minmax(name, S['fmax_vv'], S['redmax'], "-1.7976931348623157e308")
def emit_min(name): return _emit_minmax(name, S['fmin_vv'], S['redmin'], "1.7976931348623157e308")


EMITTERS = {"axpy": emit_axpy, "scal": emit_scal, "copy": emit_copy, "dot": emit_dot,
            "asum": emit_asum, "nrm2": emit_nrm2, "axpby": emit_axpby, "swap": emit_swap,
            "max": emit_max, "min": emit_min}


def emit_l1(spec: L1Spec) -> str:
    return EMITTERS[spec.kind](spec.name)


if __name__ == "__main__":
    for k in EMITTERS:
        print("// ----", k); print(emit_l1(L1Spec(name=f"d{k}_k_rvv", dtype="f64", kind=k))[:200])
