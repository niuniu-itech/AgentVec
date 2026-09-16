"""Export registered AscendC projects with contract, schedule and build records."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import uuid

from .guard import guard
from .lowering import Sched, emit_project
from .registry import ALL_OPS, recover
from .schedule import check_shape


def cases_for(air, sched, profile):
    pattern = air.s_algo.pattern.value
    if pattern in ("map", "reduce"):
        return [[sched.block_dim * sched.tile_len], [sched.block_dim * sched.tile_len * 33],
                [16777216 if profile == "paper" else 1048576]]
    if pattern == "gemv":
        return [[64, 64], [128, 192], [4096, 4096] if profile == "paper" else [512, 512]]
    return [[64, 64, 64], [128, 128, 192], [512, 512, 512] if profile == "paper" else [256, 256, 256]]


def export(op, root, sched, profile="smoke", rounds=3, iterations=10, air=None):
    air = recover(op) if air is None else air
    if air.name != op:
        raise ValueError("proposal identity differs from the registered source")
    accepted, reason = guard(air)
    record = dict(op=op, status="CHECKED" if accepted else "VETO", reason=reason,
                  intent_source="registered cuBLAS contract", schedule=asdict(sched))
    if not accepted:
        return record
    cases = cases_for(air, sched, profile)
    for case in cases:
        check_shape(air, sched, case, iterations)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    files = emit_project(air, sched)
    for name, source in files.items():
        (root / name).write_text(source, encoding="utf-8", newline="\n")
    (root / "air.json").write_text(air.to_json() + "\n", encoding="utf-8")
    project = dict(op=op, profile=profile, cases=cases, rounds=rounds, iterations=iterations,
                   schedule=asdict(sched), status="CHECKED")
    (root / "project.json").write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("runner.py"), root / "runner.py")
    record["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in (*files, "project.json")}
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", default=",".join(ALL_OPS))
    parser.add_argument("--backend", choices=("manual", "siliconflow"), default="manual")
    parser.add_argument("--model", default=os.environ.get("SILICONFLOW_MODEL"))
    parser.add_argument("--proposal", type=Path, help="one operator's s_algo JSON; contracts remain caller-owned")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--execute", choices=("local", "ascend"))
    parser.add_argument("--setup", default=os.environ.get("AGENTVEC_ASCEND_SETUP", "/usr/local/Ascend/ascend-toolkit/set_env.sh"))
    parser.add_argument("--cmake", default=os.environ.get("AGENTVEC_ASCEND_CMAKE", "cmake"))
    args = parser.parse_args(argv)
    ops = args.ops.split(",")
    if len(ops) != len(set(ops)) or not ops or set(ops) - set(ALL_OPS):
        parser.error("ops must be unique registered Ascend operator names")
    if args.backend == "siliconflow" and (not args.model or args.proposal):
        parser.error("SiliconFlow requires a model and excludes --proposal")
    if args.proposal and len(ops) != 1:
        parser.error("--proposal applies to exactly one --ops value")
    if not 1 <= args.rounds <= 100:
        parser.error("rounds must be in 1..100")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    report = dict(run_id=uuid.uuid4().hex, profile=args.profile, cases={})
    schedule = Sched(args.blocks, args.tile, args.buffers)
    for op in ops:
        try:
            directory = root / op
            air, response = None, None
            if args.proposal:
                from .proposer import from_proposal
                air = from_proposal(op, json.loads(args.proposal.read_text(encoding="utf-8")))
            elif args.backend == "siliconflow":
                from .proposer import propose
                air, response = propose(op, args.model)
            record = export(op, directory, schedule, args.profile, args.rounds, args.iterations, air=air)
            record['intent_source'] = 'supplied proposal' if args.proposal else args.backend
            record['model'] = args.model if args.backend == 'siliconflow' else None
            if response is not None:
                (root / (op+'.response.txt')).write_text(response,encoding='utf-8')
            if args.execute and record["status"] == "CHECKED":
                if args.execute == "local":
                    from .runner import run
                    tested = run(directory, args.setup, args.cmake)
                else:
                    from ..remote import put, get, run
                    remote = os.environ.get("AGENTVEC_REMOTE_ROOT", "/tmp/agentvec").rstrip("/") + "/ascend_" + report["run_id"] + "/" + op
                    for path in directory.iterdir():
                        if path.is_file():
                            put("ascend", str(path), remote + "/" + path.name)
                    command = ["python3", remote + "/runner.py", "--root", remote, "--setup", args.setup, "--cmake", args.cmake]
                    rc, stdout, stderr = run("ascend", shlex.join(command), timeout=900)
                    (directory / "remote.log").write_text(stdout + stderr, encoding="utf-8")
                    tested = json.loads(stdout)
                    if rc:
                        tested["verified"] = False
                    logs = ["build.log", "validation.json"] + [s["log"] for s in tested.get("samples", [])]
                    for log in logs:
                        try:
                            get("ascend", remote + "/" + log, str(directory / log))
                        except FileNotFoundError:
                            pass
                good = tested.get("verified") is True and tested.get("source_sha256") == record["source_sha256"]
                record.update(status="VERIFIED" if good else "FAILED", validation=tested)
            report["cases"][op] = record
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            report["cases"][op] = dict(status="FAILED", reason=str(exc))
        print(op + ": " + report["cases"][op]["status"], flush=True)
        (root / "migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(any(r["status"] == "FAILED" for r in report["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
