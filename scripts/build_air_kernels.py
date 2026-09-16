"""Compatibility entry for experiments.drivers.build_air_kernels."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "src")]

if __name__ == "__main__":
    runpy.run_module("experiments.drivers.build_air_kernels", run_name="__main__")
