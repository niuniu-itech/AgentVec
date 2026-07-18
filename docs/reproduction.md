# Reproduction Guide

## Dependencies

The host-side tools used by the RVV differential suite are:

- Python 3.10 or later and `paramiko` for optional remote execution;
- an RVV-capable cross compiler, such as `riscv64-linux-gnu-gcc`;
- `qemu-riscv64` with RVV 1.0 support, or an RVV-capable machine for native runs.

No LLM credential is needed to replay the checked manual-intent path. A fresh
model call requires `SF_KEY`; it is deliberately not stored in this artifact.

## Core modules

`src/agentvec/air.py` and `air_from_formula.py` define and construct AIR.
`guard.py` checks the declared intent. `lowering.py` emits RVV code only after a
passing guard result. `pipeline.py` combines these steps and uploads a candidate
to a configured build host for compilation and differential testing.

`remote.py` uses the following optional variables:

```text
AGENTVEC_SERVER_HOST       build host
AGENTVEC_SERVER_USER       build-host user
AGENTVEC_SERVER_KEY        SSH private-key path
AGENTVEC_SERVER_PASSWORD   alternative to key-based authentication
AGENTVEC_SERVER_GCC        RVV cross compiler
AGENTVEC_SERVER_QEMU       QEMU RVV binary
AGENTVEC_REMOTE_ROOT       remote test-suite directory
```

## Differential testing

Each package in `tests/rvv_difftest/kernels/` contains a scalar oracle, an RVV
candidate, and a `spec.json` file. `runner.py` compiles both forms, runs the
configured input sizes and seeds over each requested vector length, and emits a
JSON report. The negative control is expected to fail, confirming that the
harness detects a wrong output.

```bash
python tests/rvv_difftest/runner.py \
  --root tests/rvv_difftest \
  --gcc riscv64-linux-gnu-gcc \
  --qemu qemu-riscv64 \
  --vlens 128,256,512 \
  --seeds 5 --only saxpy --neg-control
```

## Results and scripts

`results/rvv/` contains frozen JSON summaries for the reported RVV studies.
`scripts/` retains the paper experiment drivers, including source-stress,
reasoning-mode, GEMV, and GEMM studies. These drivers assume access to the same
OpenBLAS source tree and RVV environment used in the experiments; use the core
pipeline and differential harness above for a self-contained starting point.
