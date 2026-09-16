"""Lower an ASUM intent to a CHECKED candidate and save it outside the checkout."""
import argparse
from pathlib import Path
from agentvec.air import CPhy, Dep
from agentvec.air_from_formula import air_from_intent
from agentvec.lowering import emit_rvv_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    intent = {"pattern": "reduce", "dtype": "f32", "reduce_op": "sum", "elem": "abs(a)"}
    air = air_from_intent(intent, constraints=CPhy("f32", dep=Dep.REDUCTION))
    code = emit_rvv_kernel(air)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "air.json").write_text(air.to_json(), encoding="utf-8")
    (output / "candidate.c").write_text(code, encoding="utf-8", newline="\n")
    print("CHECKED candidate saved; use agentvec migrate for independent differential testing")


if __name__ == "__main__":
    main()
