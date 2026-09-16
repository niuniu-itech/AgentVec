#!/usr/bin/env python3
"""Source-checkout entry point for the packaged differential runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agentvec.difftest import main

if __name__ == "__main__":
    sys.exit(main())
