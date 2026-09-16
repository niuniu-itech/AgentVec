"""Compile an exported DAG and run its independent validation harness."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def run(root, gcc="gcc", target="rvv", qemu="native", vlen=128):
    root = Path(root)
    files = ("candidate.c", "reference.c", "harness.c")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    command = [gcc, "-O2", "-std=c11", "-ffp-contract=off", "-fno-fast-math", "-fno-tree-vectorize"]
    if target == "rvv":
        command += ["-march=rv64gcv", "-mabi=lp64d"]
    command += [*files, "-lm", "-o", "validate_dag"]
    build = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=180)
    (root / "build.log").write_text(build.stdout + build.stderr, encoding="utf-8")
    result = dict(status="BUILD_FAILED", verified=False, source_sha256=hashes,
                  compiler_command=command, machine=platform.machine(), build_exit=build.returncode)
    if build.returncode == 0:
        executable = str((root / "validate_dag").resolve())
        launch = [executable] if qemu == "native" else [qemu, "-cpu", f"rv64,v=true,vlen={vlen},vext_spec=v1.0", executable]
        proc = subprocess.run(launch, cwd=root, capture_output=True, text=True, timeout=180)
        (root / "run.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        result.update(status="DIFF_FAILED", run_exit=proc.returncode, runner=launch)
        try:
            stats = json.loads(proc.stdout)
            passed = (proc.returncode == 0 and stats["cases"] == stats["planned_cases"] > 0
                      and stats["assertions"] > 0 and stats["failures"] == 0)
            result.update(statistics=stats, verified=passed, status="VERIFIED" if passed else "DIFF_FAILED")
        except (ValueError, KeyError, TypeError):
            result["status"] = "INVALID_RESULT"
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gcc", default="gcc")
    parser.add_argument("--target", choices=("scalar", "rvv"), default="rvv")
    parser.add_argument("--qemu", default="native")
    parser.add_argument("--vlen", type=int, choices=(128, 256, 512), default=128)
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.gcc, args.target, args.qemu, args.vlen)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = dict(status="FAILED", verified=False, reason=str(exc))
        (args.root / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
