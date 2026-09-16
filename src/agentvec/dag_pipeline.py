"""Import a DAG proposal, check its contract, lower and optionally validate it."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

from .dag import DagAIR, DagContract
from .dag_examples import EXAMPLES, example, harness_source, reference_source
from .dag_lowering import DagSchedule, emit_dag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=EXAMPLES)
    parser.add_argument("--proposal", type=Path, help="JSON DAG; never substitutes a canned answer")
    parser.add_argument("--contract", type=Path, help="independent input/output and numerical contract")
    parser.add_argument("--backend", choices=("manual", "siliconflow"), default="manual")
    parser.add_argument("--model", default=os.environ.get("SILICONFLOW_MODEL"))
    parser.add_argument("--target", choices=("rvv", "scalar"), default="rvv")
    parser.add_argument("--fuse-maps", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", choices=("local", "board", "server"))
    parser.add_argument("--gcc", default="gcc")
    parser.add_argument("--qemu", default="native")
    parser.add_argument("--vlen", type=int, choices=(128, 256, 512), default=128)
    args = parser.parse_args(argv)
    if bool(args.example) == bool(args.contract):
        parser.error("choose either --example or --contract")
    if args.contract and (not args.proposal or args.backend != "manual"):
        parser.error("custom contracts require --proposal with backend=manual")
    if args.execute and not args.example:
        parser.error("custom DAGs need an independent harness; export and validate externally")
    if args.backend == "siliconflow" and (not args.model or args.proposal):
        parser.error("SiliconFlow requires a model and excludes a supplied proposal")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    report = dict(run_id=uuid.uuid4().hex, status="VETO", example=args.example, backend=args.backend)
    try:
        if args.example:
            canonical = example(args.example)
            contract = canonical.contract
            proposal = dict(pattern="dag", nodes=[asdict(node) for node in canonical.nodes])
        else:
            contract = DagContract.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
        if args.proposal:
            proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
        elif args.backend == "siliconflow":
            from .llm_backend import chat, extract_json
            prompt = (
                'Recover a JSON DAG from this scalar reference. Return only {"pattern":"dag","nodes":[...]}. '
                'Each node has name, pattern (map or reduce), formula, and reduce_op for reductions (sum/min/max). '
                'Expressions use named inputs/intermediates, n, finite constants, +,-,*,/,exp,sqrt,abs,min,max. '
                'Inputs have these names and extents: ' + json.dumps(contract.inputs) + '. Output is y. '
                'The reference uses inputs[0], inputs[1], inputs[2] in that order. '
                'Do not return contract, status or verification claims.\n' + reference_source(args.example))
            reply = chat(args.model, prompt, max_tokens=2400)
            (root / "response.txt").write_text(reply, encoding="utf-8")
            proposal = extract_json(reply)
        graph = DagAIR.from_proposal(proposal, contract)
        source, lowering = emit_dag(graph, args.target, DagSchedule(args.fuse_maps))
        (root / "candidate.c").write_text(source, encoding="utf-8", newline="\n")
        (root / "air.json").write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")
        report.update(lowering)
        if args.example:
            (root / "reference.c").write_text(reference_source(args.example), encoding="utf-8", newline="\n")
            (root / "harness.c").write_text(harness_source(contract), encoding="utf-8", newline="\n")
            shutil.copyfile(Path(__file__).with_name("dag_runner.py"), root / "runner.py")
        expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.c")}
        report["source_sha256"] = expected
        if args.execute:
            if args.execute == "local":
                from .dag_runner import run
                validation = run(root, args.gcc, args.target, args.qemu, args.vlen)
            else:
                from .remote import get, put, run
                remote = os.environ.get("AGENTVEC_REMOTE_ROOT", "/tmp/agentvec").rstrip("/") + "/dag_" + report["run_id"]
                for p in root.iterdir():
                    if p.is_file():
                        put(args.execute, str(p), remote + "/" + p.name)
                command = ["python3", remote + "/runner.py", "--root", remote, "--target", args.target,
                           "--gcc", args.gcc, "--qemu", args.qemu, "--vlen", str(args.vlen)]
                rc, stdout, stderr = run(args.execute, shlex.join(command), timeout=400)
                (root / "remote.log").write_text(stdout + stderr, encoding="utf-8")
                validation = json.loads(stdout)
                if rc != 0:
                    validation["verified"] = False
                for name in ("build.log", "run.log", "validation.json"):
                    try:
                        get(args.execute, remote + "/" + name, str(root / name))
                    except FileNotFoundError:
                        pass
            passed = validation.get("verified") is True and validation.get("source_sha256") == expected
            report.update(status="VERIFIED" if passed else "DIFF_FAILED", validation=validation)
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        report.update(status="VETO" if report["status"] == "VETO" else "FAILED", reason=str(exc))
    (root / "migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "example", "backend")}))
    return 0 if report["status"] in ("CHECKED", "VERIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
