"""Summarize new migration records with explicit per-kernel and per-check counts."""
import argparse
import csv
import json
from pathlib import Path


def summarize(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = list(record["cases"].values())
    passed = sum(case.get("test", {}).get("passed", 0) for case in cases)
    total = sum(case.get("test", {}).get("total", 0) for case in cases)
    return dict(run_id=record["run_id"], backend=record["backend"], model=record.get("model"),
                kernels=len(cases), verified=sum(case["status"] == "VERIFIED" for case in cases),
                vetoed=sum(case["status"] == "VETO" for case in cases),
                checked_only=sum(case["status"] == "CHECKED" for case in cases),
                passed_checks=passed, executed_checks=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.records]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
