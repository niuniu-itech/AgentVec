# AgentVec

AgentVec is a research artifact for validated migration of numerical kernels to
RISC-V Vector (RVV). It separates LLM-based intent inference from deterministic
validation, lowering, and differential testing. The artifact accompanies the
paper *AgentVec: Validated Intent-Driven Migration for RISC-V Vector via
Agentic-Symbolic Alignment*.

## Repository layout

```text
src/agentvec/          AIR construction, validation, lowering, and LLM backends
scripts/               Experiment drivers used for the paper studies
tests/rvv_difftest/    Differential-test harness and RVV kernel packages
results/rvv/           Frozen JSON summaries used to prepare paper tables/figures
docs/                  Reproduction and environment notes
```

The repository contains no API keys, passwords, or host-specific configuration.
Cached model responses and generated build products are intentionally excluded.

## Quick start

Python 3.10+ is required. The differential tests additionally require an RVV
cross compiler and QEMU, or an RVV-capable host for native execution.

```bash
git clone https://github.com/niuniu-itech/AgentVec.git
cd AgentVec
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Check the Python sources:

```bash
python -m compileall -q src/agentvec tests/rvv_difftest
```

Run a small RVV differential test under QEMU:

```bash
python tests/rvv_difftest/runner.py \
  --root tests/rvv_difftest \
  --gcc riscv64-linux-gnu-gcc \
  --qemu qemu-riscv64 \
  --seeds 5 --only saxpy --neg-control
```

The full suite uses the same command without `--only saxpy`. It is substantially
larger and should be run on a machine with a suitable RVV toolchain.

## AgentVec pipeline

The default pipeline uses the checked canonical intents and requires a build host
or board configured through environment variables:

```bash
export AGENTVEC_SERVER_HOST=<host>
export AGENTVEC_SERVER_USER=<user>
export AGENTVEC_SERVER_KEY=~/.ssh/id_ed25519
export AGENTVEC_SERVER_GCC=/path/to/riscv64-linux-gnu-gcc
export AGENTVEC_SERVER_QEMU=/path/to/qemu-riscv64
export AGENTVEC_REMOTE_ROOT=/tmp/agentvec/difftest

PYTHONPATH=src/agentvec python src/agentvec/pipeline.py \
  --backend manual --ops saxpy,scal,copy,dot,asum,nrm2
```

For fresh intent proposals through the SiliconFlow-compatible backend, set
`SF_KEY` in the environment and choose `--backend siliconflow --model GLM-5.2`.
Keys are read only at runtime and are not written to the repository.

See [docs/reproduction.md](docs/reproduction.md) for the complete execution
path and the role of each directory.

## License

MIT. See [LICENSE](LICENSE).
