"""Machine-independent paths for research drivers; outputs require an external workspace."""
import os
from pathlib import Path

value = os.environ.get("AGENTVEC_EXPERIMENT_DIR")
if not value:
    raise RuntimeError("Set AGENTVEC_EXPERIMENT_DIR to an external prepared experiment directory; see experiments/README.md")
RUN_ROOT = Path(value).expanduser().resolve()
DIFFTEST_ROOT = RUN_ROOT / "difftest"
EXTRACT_ROOT = os.environ.get("AGENTVEC_EXTRACT_ROOT", "/tmp/agentvec/extract")
REMOTE_ROOT = os.environ.get("AGENTVEC_REMOTE_ROOT", "/tmp/agentvec")
BOARD_ROOT = os.environ.get("AGENTVEC_BOARD_ROOT", "/tmp/agentvec")
