"""Enumerate admitted GEMM tuples and validate an identifier-only ranking."""
import argparse
from itertools import product
import json
from pathlib import Path
from .symbolic_align import H_K1, I_GEMM, align


def enumerate_schedules(hardware=None, intent=None):
    hardware = dict(H_K1 if hardware is None else hardware)
    intent = dict(I_GEMM if intent is None else intent)
    admitted = []
    for lmul, pack, rows, block in product((1, 2, 4, 8), (True, False), (2, 4, 6, 8), (128, 256, 512)):
        config = dict(L=lmul, pack=pack, MR=rows, KC=block)
        if align(config, intent, hardware)[0]:
            admitted.append(dict(id=len(admitted), schedule=config))
    return admitted


def validate_ranking(ranked_ids, admitted_ids):
    """Ignore invented/duplicate IDs and append omitted admitted IDs deterministically."""
    if not isinstance(ranked_ids, list):
        raise ValueError("ranked_ids must be a list")
    allowed = list(admitted_ids)
    if len(set(allowed)) != len(allowed):
        raise ValueError("admitted IDs must be unique")
    result, ignored = [], []
    for identifier in ranked_ids:
        if type(identifier) is int and identifier in allowed and identifier not in result:
            result.append(identifier)
        else:
            ignored.append(identifier)
    omitted = [identifier for identifier in allowed if identifier not in result]
    return dict(order=result + omitted, ignored=ignored, appended=omitted)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", help="JSON hardware profile")
    parser.add_argument("--ranking", help="JSON with ranked_ids; no free-form schedule fields")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    hardware = json.loads(Path(args.hardware).read_text()) if args.hardware else dict(H_K1)
    candidates = enumerate_schedules(hardware)
    result = dict(mode="enumeration_only", hardware=hardware, candidates=candidates)
    if args.ranking:
        proposal = json.loads(Path(args.ranking).read_text())
        if set(proposal) - {"ranked_ids", "why_top"}:
            parser.error("a ranking cannot modify admitted schedule tuples")
        result["ranking"] = validate_ranking(proposal["ranked_ids"], [c["id"] for c in candidates])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{len(candidates)} schedules admitted; no target timing performed")
    return 0
