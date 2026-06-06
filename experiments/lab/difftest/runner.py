#!/usr/bin/env python3
"""Server-side differential-testing runner (pure stdlib, no numpy).

For each kernel: compile scalar oracle (rv64gc) and RVV variant (rv64gcv),
then for every (size, seed, VLEN) run both under qemu-riscv64 and compare
outputs with a tiered criterion (integer => bit-exact, float => rtol/ULP).
Emits a JSON summary on stdout. This operationalizes the paper's SPR metric
and is the independent oracle used by E4 (Guard validation).
"""
import json, os, struct, subprocess, sys, argparse

def sh(cmd, timeout=120):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def read_vals(path, is_float):
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    fmt = f"<{n}{'f' if is_float else 'i'}"
    return list(struct.unpack(fmt, raw))

def compare(xs, ys, is_float, rtol, atol=1e-6):
    if len(xs) != len(ys):
        return False, f"len {len(xs)}!={len(ys)}"
    for i, (x, y) in enumerate(zip(xs, ys)):
        if is_float:
            if abs(x - y) > rtol * abs(y) + atol:
                return False, f"idx{i}: {x} vs {y} (|d|={abs(x-y):.3e})"
        else:
            if x != y:
                return False, f"idx{i}: {x} vs {y}"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gcc", required=True)
    ap.add_argument("--qemu", required=True)
    ap.add_argument("--vlens", default="128,256,512")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--neg-control", action="store_true",
                    help="also build saxpy with INJECT_BUG and confirm it FAILS")
    ap.add_argument("--only", default="", help="comma-separated substrings; only test matching kernel dirs")
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]

    root = args.root
    bd = os.path.join(root, "build"); os.makedirs(bd, exist_ok=True)
    vlens = [int(v) for v in args.vlens.split(",")]
    kdir = os.path.join(root, "kernels")
    kernels = sorted(d for d in os.listdir(kdir) if os.path.isdir(os.path.join(kdir, d)))
    if only:
        kernels = [k for k in kernels if any(s in k for s in only)]

    report = {"vlens": vlens, "seeds": args.seeds, "kernels": {}}
    for k in kernels:
        spec = json.load(open(os.path.join(kdir, k, "spec.json")))
        is_float = spec["dtype"].startswith("f")
        rtol = float(spec.get("rtol", 0.0))
        sizes = spec["sizes"]
        sc, rv = os.path.join(kdir, k, "scalar.c"), os.path.join(kdir, k, "rvv.c")
        bsc, brv = os.path.join(bd, f"{k}_scalar"), os.path.join(bd, f"{k}_rvv")
        rc1, _, e1 = sh(f"{args.gcc} -O3 -march=rv64gc  -static -I{root} {sc} -o {bsc} -lm")
        rc2, _, e2 = sh(f"{args.gcc} -O3 -march=rv64gcv -static -I{root} {rv} -o {brv} -lm")
        if rc1 or rc2:
            report["kernels"][k] = {"build_error": (e1 + e2)[:400]}
            continue

        total = passed = 0
        first_fail = None
        for size in sizes:
            for seed in range(1, args.seeds + 1):
                os_ = os.path.join(bd, "os.bin"); orr = os.path.join(bd, "or.bin")
                sh(f"{args.qemu} {bsc} {seed} {size} {os_}")
                xs = read_vals(os_, is_float)
                for V in vlens:
                    rcq, _, eq = sh(f"{args.qemu} -cpu rv64,v=true,vlen={V},vext_spec=v1.0 {brv} {seed} {size} {orr}")
                    total += 1
                    if rcq != 0:
                        if not first_fail: first_fail = f"size{size} seed{seed} vlen{V}: qemu rc={rcq} {eq[:120]}"
                        continue
                    ys = read_vals(orr, is_float)
                    ok, why = compare(xs, ys, is_float, rtol)
                    if ok: passed += 1
                    elif not first_fail: first_fail = f"size{size} seed{seed} vlen{V}: {why}"
        report["kernels"][k] = {
            "tier": spec["tier"], "rtol": rtol, "sizes": sizes,
            "total": total, "passed": passed, "spr": round(passed / total, 4) if total else 0,
            "first_fail": first_fail,
        }

    if args.neg_control:
        # Build a deliberately-wrong saxpy; the tester MUST flag it as failing.
        sc = os.path.join(kdir, "saxpy", "scalar.c"); rv = os.path.join(kdir, "saxpy", "rvv.c")
        bsc = os.path.join(bd, "nc_scalar"); brv = os.path.join(bd, "nc_rvv_bug")
        sh(f"{args.gcc} -O3 -march=rv64gc  -static -I{root} {sc} -o {bsc}")
        sh(f"{args.gcc} -O3 -march=rv64gcv -static -DINJECT_BUG -I{root} {rv} -o {brv}")
        sh(f"{args.qemu} {bsc} 1 1000 {os.path.join(bd,'os.bin')}")
        xs = read_vals(os.path.join(bd, "os.bin"), True)
        sh(f"{args.qemu} -cpu rv64,v=true,vlen=128,vext_spec=v1.0 {brv} 1 1000 {os.path.join(bd,'or.bin')}")
        ys = read_vals(os.path.join(bd, "or.bin"), True)
        ok, why = compare(xs, ys, True, 1e-6)
        report["neg_control"] = {"buggy_kernel_detected_as_failing": (not ok), "evidence": why}

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
