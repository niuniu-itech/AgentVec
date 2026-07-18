"""Generate difftest kernels whose RVV side is AIR-lowered and whose scalar
oracle is the INDEPENDENT hand-written reference. Validates Phase-3 lowering."""
import os, shutil, json
from air import AIR, SAlgo, CPhy, MMap, Expr, Pattern as P, Dep, Alias, Access
from lowering import emit_rvv_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF = os.path.normpath(os.path.join(HERE, "..", "lab", "difftest"))
Kroot = os.path.join(DIFF, "kernels")

AIRS = {
    "saxpy": AIR(s_algo=SAlgo(pattern=P.MAP,
                 elem_op=Expr("add", args=[Expr("mulc", val=2.0, args=[Expr("load", src="a")]),
                                           Expr("load", src="b")])),
                 c_phy=CPhy(dtype="f32", dep=Dep.NONE, alias=Alias.DISJOINT),
                 m_map=MMap(access=Access.UNIT_STRIDE)),
    "iaxpy": AIR(s_algo=SAlgo(pattern=P.MAP,
                 elem_op=Expr("sub", args=[Expr("mulc", val=3.0, args=[Expr("load", src="a")]),
                                           Expr("load", src="b")])),
                 c_phy=CPhy(dtype="i32", dep=Dep.NONE, alias=Alias.DISJOINT)),
    "fsum": AIR(s_algo=SAlgo(pattern=P.REDUCE, reduce_op="sum", reduce_src="a"),
                c_phy=CPhy(dtype="f32", dep=Dep.REDUCTION, alias=Alias.DISJOINT)),
}

for name, air in AIRS.items():
    ok, msg = air.validate_schema()
    assert ok, f"{name}: {msg}"
    dst = os.path.join(Kroot, name + "_air")
    os.makedirs(dst, exist_ok=True)
    # RVV side: AIR-lowered (the thing under test)
    with open(os.path.join(dst, "rvv.c"), "w", newline="\n") as f:
        f.write(emit_rvv_kernel(air))
    # scalar oracle: INDEPENDENT hand-written reference copied from the base kernel
    shutil.copy(os.path.join(Kroot, name, "scalar.c"), os.path.join(dst, "scalar.c"))
    spec = json.load(open(os.path.join(Kroot, name, "spec.json")))
    spec["name"] = name + "_air"
    json.dump(spec, open(os.path.join(dst, "spec.json"), "w"))
    # also save the AIR instance as provenance
    open(os.path.join(dst, "air.json"), "w").write(air.to_json())
    print(f"generated {name}_air (rvv<-AIR, scalar<-hand oracle)")
print("done ->", Kroot)
