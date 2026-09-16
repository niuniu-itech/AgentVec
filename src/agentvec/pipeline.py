"""Registered-kernel migration through intent, AIR checks, lowering and difftest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

from . import llm_backend as LB
from . import obfuscate
from .air import CPhy, Dep
from .air_from_formula import air_from_intent
from .guard import guard
from .lowering import emit_rvv_kernel

MAC_R = "#include <math.h>\n#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 1\n"

# operator zoo: name -> (kind, scalar-oracle body, AgentVec lowering spec, rtol)
ZOO = {
    "saxpy": ("map",    "out[i]=2.0f*a[i]+b[i];",              ("map", "out=2*a+b"),       1e-6),
    "scal":  ("map",    "out[i]=3.0f*a[i];",                   ("map", "out=3*a"),         1e-6),
    "copy":  ("map",    "out[i]=a[i];",                        ("map", "out=a"),           0.0),
    "dot":   ("reduce", "s+=a[i]*b[i];",                       ("sum", "ab", None),        1e-4),
    "asum":  ("reduce", "s+=fabsf(a[i]);",                     ("sum", "abs_a", None),     1e-4),
    "nrm2":  ("reduce", "s+=a[i]*a[i];",                       ("sum", "aa", "sqrt"),      1e-4),
}


def oracle(op):
    kind, body, _, _ = ZOO[op]
    if kind == "map":
        return ("#define DT float\n#define DT_IS_FLOAT 1\n#define REDUCE 0\n"
                "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)c;"
                f"for(int i=0;i<n;i++){body}}}\n#include \"harness.h\"\n")
    return (MAC_R + "void agentvec_kernel(const DT*a,const DT*b,const DT*c,DT*out,int n){(void)c;(void)b;"
            f"float s=0;for(int i=0;i<n;i++){body}out[0]={'sqrtf(s)' if op=='nrm2' else 's'};}}\n#include \"harness.h\"\n")


def canonical_intent(op):
    kind, _, spec, _ = ZOO[op]
    if kind == "map":
        return dict(pattern="map", dtype="f32", formula=spec[1])
    return dict(pattern="reduce", dtype="f32", reduce_op=spec[0],
                elem={"a": "a", "ab": "a*b", "aa": "a*a", "abs_a": "abs(a)"}[spec[1]], postproc=spec[2])


def checked_air(op, intent):
    if not isinstance(intent, dict) or intent.get("pattern") != ZOO[op][0]:
        raise ValueError("proposal pattern disagrees with the registered source contract")
    contract = CPhy("f32", dep=Dep.REDUCTION if ZOO[op][0] == "reduce" else Dep.NONE)
    air = air_from_intent(intent, constraints=contract)
    ok, reason = guard(air)
    if not ok:
        raise ValueError(reason)
    return air


def lower_from_intent(op, intent):
    """Missing or malformed proposals are rejected, never replaced by an answer."""
    if intent is None:
        raise ValueError("missing intent")
    if not isinstance(intent, dict):
        if ZOO[op][0] == "map":
            intent = dict(pattern="map", dtype="f32", formula=intent)
        else:
            rop, pre, post = intent
            expressions = {"a": "a", "ab": "a*b", "abs_a": "abs(a)", "aa": "a*a"}
            if pre not in expressions:
                raise ValueError("unsupported reduction expression")
            intent = dict(pattern="reduce", dtype="f32", reduce_op=rop, elem=expressions[pre], postproc=post)
    return emit_rvv_kernel(checked_air(op, intent))


def prepare_case(op, intent, root):
    air = checked_air(op, intent)
    code = emit_rvv_kernel(air)
    directory = root / "kernels" / op
    directory.mkdir(parents=True)
    (directory / "rvv.c").write_text(code, encoding="utf-8", newline="\n")
    (directory / "scalar.c").write_text(oracle(op), encoding="utf-8", newline="\n")
    spec = dict(name=op, dtype="f32", reduce=ZOO[op][0] == "reduce", tier="relative_absolute",
                rtol=ZOO[op][3], atol=1e-6, sizes=[0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 1000])
    (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (directory / "air.json").write_text(air.to_json(), encoding="utf-8")
    return dict(status="CHECKED", intent=intent, proof=air.v_meta.proof,
                source_sha256=hashlib.sha256(oracle(op).encode()).hexdigest(),
                candidate_sha256=hashlib.sha256(code.encode()).hexdigest(),
                spec_sha256=hashlib.sha256((directory / "spec.json").read_bytes()).hexdigest())


def accepts_test(record, tested, exit_code):
    hashes = tested.get("source_sha256", {})
    return (exit_code in (0, 1) and tested.get("verified") is True
            and tested.get("planned", 0) > 0
            and tested.get("passed") == tested.get("total") == tested.get("planned")
            and not tested.get("build_error") and not tested.get("first_fail")
            and hashes.get("rvv.c") == record["candidate_sha256"]
            and hashes.get("scalar.c") == record["source_sha256"]
            and hashes.get("spec.json") == record["spec_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="manual", choices=["manual", "siliconflow"])
    parser.add_argument("--model", default=os.environ.get("SILICONFLOW_MODEL"))
    parser.add_argument("--ops", default=",".join(ZOO))
    parser.add_argument("--output", required=True, help="new directory for generated code and records")
    parser.add_argument("--execute", choices=["local", "server", "board"], help="omit to generate CHECKED candidates only")
    parser.add_argument("--gcc", help="compiler executable on the selected host")
    parser.add_argument("--qemu", help="qemu-riscv64 or native")
    parser.add_argument("--vlens", default="128,256,512")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--obfuscate", action="store_true")
    args = parser.parse_args(argv)
    ops = [item.strip() for item in args.ops.split(",")]
    if not ops or set(ops) - set(ZOO) or len(ops) != len(set(ops)):
        parser.error("ops must contain unique registered kernel names")
    if args.backend == "siliconflow" and not args.model:
        parser.error("SiliconFlow requires --model or SILICONFLOW_MODEL")
    if args.seeds < 1:
        parser.error("seeds must be positive")
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output directory must be empty; use a new run directory")
    root.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parent
    shutil.copyfile(package / "data" / "harness.h", root / "harness.h")
    shutil.copyfile(package / "difftest.py", root / "runner.py")
    report = dict(run_id=uuid.uuid4().hex, backend=args.backend, model=args.model if args.backend != "manual" else None,
                  scope="registered unit-stride f32 map/reduce", cases={})
    for op in ops:
        try:
            if args.backend == "manual":
                intent = canonical_intent(op)
            else:
                source = oracle(op)
                if args.obfuscate:
                    source = obfuscate.obfuscate(source)
                reply = LB.chat(LB.MODELS.get(args.model, args.model), LB.AIR_PROMPT.format(src=source), max_tokens=1200)
                intent = LB.extract_json(reply)
            report["cases"][op] = prepare_case(op, intent, root)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            report["cases"][op] = dict(status="VETO", reason=str(exc), fallback="registered scalar oracle")
        print(f"{op}: {report['cases'][op]['status']}", flush=True)
    selected = [op for op in ops if report["cases"][op]["status"] == "CHECKED"]
    if args.execute and selected:
        common = ["--seeds", str(args.seeds), "--vlens", args.vlens, "--exact", ",".join(selected)]
        if args.execute == "local":
            command = [sys.executable, str(root / "runner.py"), "--root", str(root),
                       "--gcc", args.gcc or "gcc", "--qemu", args.qemu or "native", *common]
            process = subprocess.run(command, capture_output=True, text=True)
            rc, stdout, stderr = process.returncode, process.stdout, process.stderr
        else:
            from .remote import HOSTS, put, run
            host = args.execute
            remote = os.environ.get("AGENTVEC_REMOTE_ROOT", "/tmp/agentvec").rstrip("/") + "/" + report["run_id"]
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    put(host, str(file), remote + "/" + file.relative_to(root).as_posix())
            gcc = args.gcc or (HOSTS[host]["riscv_gcc"] if host == "server" else "gcc")
            qemu = args.qemu or (HOSTS[host]["qemu"] if host == "server" else "native")
            command = ["python3", remote + "/runner.py", "--root", remote, "--gcc", gcc, "--qemu", qemu, *common]
            rc, stdout, stderr = run(host, shlex.join(command), timeout=1800)
        (root / "difftest.stdout.json").write_text(stdout, encoding="utf-8")
        (root / "difftest.stderr.txt").write_text(stderr, encoding="utf-8")
        report["runner_exit_code"] = rc
        try:
            result = json.loads(stdout)
            for op in selected:
                tested = result["kernels"][op]
                record = report["cases"][op]
                record["status"] = "VERIFIED" if accepts_test(record, tested, rc) else "REJECTED"
                record["test"] = tested
                if record["status"] != "VERIFIED":
                    record["fallback"] = "registered scalar oracle"
        except (ValueError, KeyError, TypeError):
            for op in selected:
                report["cases"][op].update(status="EXECUTION_ERROR", reason="missing or invalid runner report")
    report["verified_count"] = sum(record["status"] == "VERIFIED" for record in report["cases"].values())
    (root / "migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        return 0 if report["verified_count"] == len(ops) else 1
    return 0 if len(selected) == len(ops) else 1


if __name__ == "__main__":
    sys.exit(main())
