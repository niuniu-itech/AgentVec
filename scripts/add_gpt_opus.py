"""Compatibility entry for experiments.drivers.add_gpt_opus."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "src")]

if __name__ == "__main__":
    runpy.run_module("experiments.drivers.add_gpt_opus", run_name="__main__")
