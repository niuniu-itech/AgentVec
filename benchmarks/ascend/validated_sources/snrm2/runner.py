"""Build an exported AscendC project and retain every target measurement."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess


def parse_result(stdout, op, sizes):
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT ")]
    if len(lines) != 1:
        raise ValueError("expected exactly one result line")
    fields = dict(item.split("=", 1) for item in lines[0].split()[1:] if "=" in item)
    if fields.get("op") != op or fields.get("pass") != "1":
        raise ValueError("wrong operator or failed differential check")
    keys = ("n",) if len(sizes) == 1 else ("m", "n") if len(sizes) == 2 else ("m", "n", "k")
    if [int(fields[key]) for key in keys] != sizes:
        raise ValueError("executed dimensions differ from the requested workload")
    for key in ("us", "maxrel"):
        if key not in fields or not math.isfinite(float(fields[key])) or float(fields[key]) < 0:
            raise ValueError("invalid timing or error statistic")
    if float(fields["us"]) == 0:
        raise ValueError("latency must be positive")
    return fields


def run(root, setup, cmake="cmake", timeout=600):
    root = Path(root).resolve()
    spec = json.loads((root / "project.json").read_text(encoding="utf-8"))
    files = (f"{spec['op']}.cpp", "main.cpp", "CMakeLists.txt", "project.json")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    result = dict(status="BUILD_FAILED", verified=False, source_sha256=hashes,
                  machine=platform.machine(), timing_boundary="host_batch_submit_sync",
                  reduction_finalization="CPU partial fold outside timed kernel batch",
                  samples=[], planned=len(spec["cases"]) * spec["rounds"])
    prefix = "set -e; source " + shlex.quote(setup) + "; "
    build = root / "build"
    config = [cmake, "-S", str(root), "-B", str(build), "-DRUN_MODE=npu", "-DSOC_VERSION=Ascend310P1",
              "-DCMAKE_BUILD_TYPE=Release"]
    toolkit = os.environ.get("ASCEND_CANN_PACKAGE_PATH")
    if toolkit:
        config.append("-DASCEND_CANN_PACKAGE_PATH=" + toolkit)
    command = prefix + shlex.join(config) + " && " + shlex.join([cmake, "--build", str(build), "-j2"])
    proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
    (root / "build.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    result["build_exit"] = proc.returncode
    if proc.returncode == 0:
        binaries = [p for p in build.rglob(spec["op"] + "_bbit") if p.is_file() and os.access(p, os.X_OK)]
        if len(binaries) != 1:
            result["reason"] = "expected one built benchmark executable"
        else:
            result["status"] = "DIFF_FAILED"
            for case in spec["cases"]:
                for iteration in range(spec["rounds"]):
                    args = [str(binaries[0]), *map(str, case), str(spec["iterations"])]
                    measured = subprocess.run(["bash", "-c", prefix + shlex.join(args)],
                                              capture_output=True, text=True, timeout=timeout)
                    log = f"case{len(result['samples']):03d}.log"
                    (root / log).write_text(measured.stdout + measured.stderr, encoding="utf-8")
                    sample = dict(sizes=case, round=iteration, exit_code=measured.returncode, log=log, passed=False)
                    try:
                        if measured.returncode:
                            raise ValueError("target program returned failure")
                        sample.update(metrics=parse_result(measured.stdout, spec["op"], case), passed=True)
                    except (KeyError, ValueError) as exc:
                        sample["reason"] = str(exc)
                    result["samples"].append(sample)
            result["verified"] = (len(result["samples"]) == result["planned"] > 0
                                  and all(sample["passed"] for sample in result["samples"]))
            result["status"] = "VERIFIED" if result["verified"] else "DIFF_FAILED"
            result["medians_us"] = {
                "x".join(map(str, case)): statistics.median(float(s["metrics"]["us"]) for s in result["samples"]
                                                          if s["sizes"] == case and s["passed"])
                for case in spec["cases"] if any(s["sizes"] == case and s["passed"] for s in result["samples"])}
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--setup", default=os.environ.get("AGENTVEC_ASCEND_SETUP", "/usr/local/Ascend/ascend-toolkit/set_env.sh"))
    parser.add_argument("--cmake", default=os.environ.get("AGENTVEC_ASCEND_CMAKE", "cmake"))
    args = parser.parse_args(argv)
    try:
        record = run(args.root, args.setup, args.cmake)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record = dict(status="FAILED", verified=False, reason=str(exc))
        (args.root / "validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))
    return 0 if record["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
