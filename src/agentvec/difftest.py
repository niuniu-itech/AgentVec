"""Compile and compare independent scalar/RVV kernels on QEMU or a native board.

This module is also a standalone runner for remote hosts with only Python.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def read_vals(path, is_float, dtype=None):
    dtype = dtype or ("f32" if is_float else "i32")
    code = {"f32": "f", "f64": "d", "i32": "i"}[dtype]
    raw = Path(path).read_bytes()
    width = struct.calcsize(code)
    if len(raw) % width:
        raise ValueError("truncated output file")
    return list(struct.unpack(f"<{len(raw) // width}{code}", raw))


def ulp_distance(x, y, dtype="f32"):
    floating, integer = ("f", "I") if dtype == "f32" else ("d", "Q")
    sign = 1 << (struct.calcsize(integer) * 8 - 1)

    def ordered(value):
        bits = struct.unpack("<" + integer, struct.pack("<" + floating, value))[0]
        return sign - (bits & (sign - 1)) if bits & sign else sign + bits

    return abs(ordered(x) - ordered(y))


def compare(reference, candidate, is_float, rtol, atol=1e-6, max_ulps=None, dtype="f32"):
    if any(not math.isfinite(t) or t < 0 for t in (rtol, atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    if max_ulps is not None and (type(max_ulps) is not int or max_ulps < 0):
        raise ValueError("max_ulps must be a nonnegative integer")
    if len(reference) != len(candidate):
        return False, "output length mismatch"
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if is_float:
            if math.isnan(expected) or math.isnan(actual):
                return False, f"index {index}: NaN outside the comparison domain"
            if expected == actual:
                continue
            if not math.isfinite(expected) or not math.isfinite(actual):
                return False, f"index {index}: non-finite mismatch"
            if abs(expected - actual) > atol + rtol * abs(expected):
                return False, f"index {index}: {actual} != reference {expected}"
            if max_ulps is not None and ulp_distance(expected, actual, dtype) > max_ulps:
                return False, f"index {index}: ULP budget exceeded"
        elif expected != actual:
            return False, f"index {index}: {actual} != reference {expected}"
    return True, ""


def _execute(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"execution exceeded {timeout} seconds"
    except OSError as exc:
        return 127, "", str(exc)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_length(spec, scalar_source, size):
    """Use the independent reference's harness ABI for legacy multi-output cases."""
    def define(name, default):
        match = re.search(r"^\s*#\s*define\s+" + name + r"\s+(\d+)\s*$", scalar_source, re.M)
        return int(match.group(1)) if match else default
    if spec.get("reduce"):
        return 1
    matrix = spec.get("out_is_matrix", define("OUT_IS_MATRIX", 0))
    factor = spec.get("out_factor", define("OUT_FACTOR", 1))
    return size * size if matrix else size * factor


def evaluate_kernel(directory, build, header_root, gcc, qemu, vlens, seeds, timeout,
                    fail_fast=True, injected=False):
    spec = json.loads((directory / "spec.json").read_text())
    dtype = spec["dtype"]
    if dtype not in ("f32", "f64", "i32"):
        raise ValueError(f"unsupported dtype {dtype}")
    sizes = spec["sizes"]
    if not sizes or any(type(n) is not int or n < 0 for n in sizes):
        raise ValueError("sizes must be nonnegative integers")
    build.mkdir(parents=True, exist_ok=True)
    native = qemu in ("native", "none", "")
    binaries = [build / "scalar", build / "rvv"]
    report = dict(dtype=dtype, sizes=sizes, seeds=seeds, vlens=vlens,
                  source_sha256={name: _sha(directory / name) for name in ("scalar.c", "rvv.c", "spec.json")},
                  rtol=spec.get("rtol", 0), atol=spec.get("atol", 1e-6), max_ulps=spec.get("max_ulps"),
                  total=0, passed=0, verified=False)
    for source, binary, arch in zip(("scalar.c", "rvv.c"), binaries, ("rv64gc", "rv64gcv")):
        flags = ["-O3", "-ffp-contract=off", "-fno-tree-vectorize", f"-march={arch}"]
        if not native:
            flags += ["-static"]
        if injected and source == "rvv.c":
            flags += ["-DINJECT_BUG"]
        command = [gcc, *flags, "-I", str(header_root), str(directory / source), "-o", str(binary), "-lm"]
        code, _, error = _execute(command, 120)
        if code:
            report.update(build_error=error[:2000], build_exit_code=code)
            return report
    outputs = [build / "scalar.bin", build / "rvv.bin"]
    first_failure = None
    for size in sizes:
        for seed in range(1, seeds + 1):
            for vlen in vlens:
                values = []
                failure = None
                for i, binary in enumerate(binaries):
                    outputs[i].unlink(missing_ok=True)
                    command = [str(binary), str(seed), str(size), str(outputs[i])]
                    if not native:
                        prefix = [qemu]
                        if i == 1:
                            prefix += ["-cpu", f"rv64,v=true,vlen={vlen},vext_spec=v1.0"]
                        command = prefix + command
                    code, _, error = _execute(command, timeout)
                    if code:
                        failure = f"{'candidate' if i else 'reference'} exit {code}: {error[:250]}"
                        break
                    if not outputs[i].exists():
                        failure = "executable produced no output file"
                        break
                    try:
                        values.append(read_vals(outputs[i], dtype.startswith("f"), dtype))
                    except ValueError as exc:
                        failure = str(exc)
                        break
                if failure is None:
                    expected = output_length(spec, (directory / "scalar.c").read_text(), size)
                    if len(values[0]) != expected:
                        failure = f"reference length {len(values[0])} != specified {expected}"
                    else:
                        passed, why = compare(*values, dtype.startswith("f"), report["rtol"],
                                              report["atol"], report["max_ulps"], dtype)
                        if not passed:
                            failure = why
                report["total"] += 1
                if failure:
                    first_failure = first_failure or dict(size=size, seed=seed, vlen=vlen, reason=failure)
                else:
                    report["passed"] += 1
                if failure and fail_fast:
                    break
            if first_failure and fail_fast:
                break
        if first_failure and fail_fast:
            break
    report["first_fail"] = first_failure
    report["planned"] = len(sizes) * seeds * len(vlens)
    report["check_pass_rate"] = report["passed"] / report["total"] if report["total"] else 0
    report["verified"] = report["passed"] == report["planned"] and first_failure is None
    # Preserve the legacy field; a kernel passes only if every requested check passes.
    report["spr"] = float(report["verified"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gcc", required=True)
    parser.add_argument("--qemu", default="native")
    parser.add_argument("--vlens", default="128,256,512")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--run-timeout", type=int, default=30)
    parser.add_argument("--build-dir")
    parser.add_argument("--output")
    parser.add_argument("--only", default="", help="comma-separated name substrings")
    parser.add_argument("--exact", default="", help="comma-separated exact kernel names")
    parser.add_argument("--neg-control", action="store_true")
    parser.add_argument("--no-fail-fast-kernel", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.run_timeout < 1:
        parser.error("seeds and timeout must be positive")
    native = args.qemu in ("native", "none", "")
    vlens = ["native"] if native else [int(v) for v in args.vlens.split(",")]
    if not native and any(v < 128 or v & (v - 1) for v in vlens):
        parser.error("VLENs must be powers of two >= 128")
    root = Path(args.root).resolve()
    build = Path(args.build_dir).resolve() if args.build_dir else root / "build"
    names = sorted(p.name for p in (root / "kernels").iterdir() if p.is_dir())
    if args.exact:
        selected = args.exact.split(",")
        if set(selected) - set(names):
            parser.error("one or more exact kernel names do not exist")
        names = sorted(set(selected))
    elif args.only:
        names = [n for n in names if any(value in n for value in args.only.split(","))]
    if not names:
        parser.error("no kernels selected")
    report = dict(mode="native" if native else "qemu", vlens=vlens, seeds=args.seeds, kernels={})
    for name in names:
        report["kernels"][name] = evaluate_kernel(root / "kernels" / name, build / name, root,
            args.gcc, args.qemu, vlens, args.seeds, args.run_timeout, not args.no_fail_fast_kernel)
    success = all(value["verified"] for value in report["kernels"].values())
    if args.neg_control:
        directory = root / "kernels" / "saxpy"
        if not directory.is_dir() or "INJECT_BUG" not in (directory / "rvv.c").read_text():
            parser.error("negative control requires saxpy with an INJECT_BUG branch")
        control = evaluate_kernel(directory, build / "negative_control", root, args.gcc, args.qemu,
                                  vlens, 1, args.run_timeout, injected=True)
        detected = (not control.get("build_error") and bool(control.get("first_fail"))
                    and "reference" in control["first_fail"]["reason"] and "!=" in control["first_fail"]["reason"])
        report["neg_control"] = dict(buggy_kernel_detected_as_failing=detected, evidence=control)
        success = success and detected
    report["success"] = success
    text = json.dumps(report, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
