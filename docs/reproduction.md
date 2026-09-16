# Reproduction guide

## CPU validation

```bash
python -m pip install -e '.[dev,remote]'
python -m pytest
```

These tests check complete expression parsing, rejected contracts, CHECKED/VERIFIED
separation, malformed model output, cache identity, identifier-only ranking, and
numerical comparison. They require no remote host or model API.

## Native registered kernels

On an RVV board, run:

```bash
agentvec migrate --backend manual --execute local --gcc gcc --qemu native \
  --seeds 5 --output /tmp/agentvec-run-001
```

For execution through SSH from another machine, set the board variables in
`.env.example` and replace `--execute local` with `--execute board`.
Generated sources and reports stay in the explicit output directory. The runner
uses a unique remote directory, checks compiler exit codes, and selects only the
current run's kernel names. Each case retains its source and test record hashes.

## Cross-VLEN validation

Use the `server` profile with a RISC-V cross compiler and QEMU, then run:

```bash
agentvec migrate --backend manual --execute server --vlens 128,256,512,1024 \
  --seeds 5 --output /external/agentvec-cross-vlen-001
```

A native run exercises one installed VLEN only. Refer to [native tests](../tests/native/README.md)
for additional tails, reduction identities and guarded NRM2 boundary checks.

For the distributed corpus, select a bounded subset first:

```bash
agentvec difftest --root tests/rvv_difftest --gcc riscv64-linux-gnu-gcc \
  --qemu qemu-riscv64 --exact saxpy --seeds 5 --neg-control \
  --build-dir /external/agentvec-corpus-build --output /external/agentvec-corpus.json
```

The negative control must compile, execute, and disagree with the reference. A
compiler failure is not evidence that an injected semantic bug was detected.

## Numerical policy and evidence

The runner accepts `dtype`, `sizes`, `reduce`, `rtol`, optional `atol`, and optional
`max_ulps` from a case specification. When `max_ulps` is present, both the ULP and
relative/absolute conditions must pass. NaNs are outside this comparator's accepted
domain; matching signed infinities are allowed for reduction identities. The
standalone guarded-norm test defines its own explicit non-finite policy.

`check_pass_rate` counts executed dynamic checks. `verified` means every planned
check passed; `spr` is its binary per-kernel compatibility field. A partially run
kernel is not accepted. Frozen legacy summaries retain their original denominators.

## Full paper studies

The study drivers require external extraction/build inputs, matching model IDs,
and the recorded experiment settings. See [experiment entry points](../experiments/README.md)
and [paper coverage](paper-coverage.md). Complete paper reproduction also needs
matching extraction inputs, model settings and hardware runs; this release does
not present a local unit test run as completion of those campaigns. [General
graphs](dag.md) and the [AscendC probe](ascend.md) now have separate generation and
validation entry points.
